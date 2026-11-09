import numpy as np
import time

def moving_sofa_3d_path() -> np.ndarray:
    """
    3D moving sofa: find a motion path through an L-shaped 3D corridor.
    
    Corridor:
      Arm 1: x<=0, 0<=y<=1, 0<=z<=1  (entry from -x)
      Arm 2: 0<=x<=1, 0<=y<=5, 0<=z<=1 (horizontal, +y)
      Arm 3: 0<=x<=1, 5<=y<=6, z>=0  (vertical, +z)
    
    Start pose (prepended): (-3, 0, 0, 0, 0, 0)
    End pose (appended): (0, 6, 7, 90, 90, 90)
    
    Strategy:
      - Keep the block near the corridor centerline
      - Turn 1 (yaw 0->90): pivot in the x-y plane at the corner
      - Turn 2 (pitch 0->90): pivot in the x-z plane at the top corner
      - Use the center of the corridor cross-section for rotation axes
    """
    # Build path segment by segment
    # Each pose: (x, y, z, yaw, pitch, roll) in degrees
    
    poses = []
    
    # Phase 1: Move from entry into arm 2, building up yaw to 90
    # At the start, the block is in arm 1 (x<=0, y in [0,1], z in [0,1])
    # We need to rotate yaw from 0 to 90 while moving into the corner region
    
    # The key insight: for a yaw rotation about z-axis, a point at (dx, dy) 
    # from the rotation center sweeps an arc. The bounding box of this arc
    # must fit in the corridor. Near the corner (x~0.5, y~0.5), the corridor
    # is 1x1 in x-y. So the rotation center should be near (0.5, 0.5).
    
    # Phase 1a: Move x from -3 toward 0, keeping y~0.5, z~0.5, yaw~0
    # Then at the corner, rotate yaw to 90 while staying in the 1x1 corner
    
    # Actually, let's think about this more carefully.
    # The evaluator prepends (-3,0,0,0,0,0) and appends (0,6,7,90,90,90).
    # Our path goes from near the start to near the end.
    
    # Let's design a smooth path:
    # 1. Move from (-3,0,0) to (0, 0.5, 0.5) with yaw 0 -> small yaw
    # 2. At the corner, rotate yaw 0->90 while staying in x∈[0,1], y∈[0,1]
    # 3. Move along arm 2 from y=1 to y=5, with yaw=90
    # 4. At the top corner, rotate pitch 0->90 while staying in x∈[0,1], y∈[5,6]
    # 5. Move along arm 3 upward with pitch=90, yaw=90
    
    # For maximum volume, we want to keep as many grid points alive as possible.
    # The grid is x in linspace(-4,0,81), y,z in linspace(0,1,21).
    # So grid points have x in [-4,0], y in [0,1], z in [0,1].
    
    # Wait - the grid is FIXED at x∈[-4,0], y∈[0,1], z∈[0,1].
    # The block is placed at R*p + t where p is a grid point.
    # So the block's extent depends on the pose.
    
    # For a grid point p=(px,py,pz) to survive:
    # At each pose (x,y,z,yaw,pitch,roll), the placed point is R*p + (x,y,z)
    # This must be in the corridor.
    
    # Since p has px in [-4,0], py,pz in [0,1], and we need R*p + t in corridor,
    # the translation t must shift the block into the corridor.
    
    # At start pose (-3,0,0,0,0,0): R=I, so placed point = p + (-3,0,0) = (px-3, py, pz)
    # px in [-4,0] -> px-3 in [-7,-3]. But corridor arm 1 is x<=0, y∈[0,1], z∈[0,1].
    # So px-3 <= 0 means px <= 3, which is always true. And y=py in [0,1], z=pz in [0,1].
    # So ALL grid points are in the corridor at the start pose! Good.
    
    # At end pose (0,6,7,90,90,90): 
    # R = Rx(90)Ry(90)Rz(90)
    # Let's compute: Rz(90) rotates x->y, y->-x
    # Ry(90) rotates z->x, x->-z
    # Rx(90) rotates y->z, z->-y
    # Combined: p=(px,py,pz) -> after Rz(90): (0, px, -py)
    # after Ry(90): (pz, px, 0)  [Ry(90): x'=z, y'=y, z'=-x on (0,px,-py) gives (−py, px, 0)? Let me redo]
    
    # Actually let me just use the rotation matrices properly.
    # Rz(yaw) = [[cos,-sin,0],[sin,cos,0],[0,0,1]]
    # Ry(pitch) = [[cos,0,sin],[0,1,0],[-sin,0,cos]]
    # Rx(roll) = [[1,0,0],[0,cos,-sin],[0,sin,cos]]
    # R = Rx @ Ry @ Rz
    
    # For yaw=90, pitch=90, roll=90:
    # Rz(90) = [[0,-1,0],[1,0,0],[0,0,1]]
    # Ry(90) = [[0,0,1],[0,1,0],[-1,0,0]]
    # Rx(90) = [[1,0,0],[0,0,-1],[0,1,0]]
    # Ry@Rz = [[0,-1,0],[0,0,1],[-1,0,0]] @ [[0,-1,0],[1,0,0],[0,0,1]] 
    #        = [[0*0+(-1)*1+0*0, 0*(-1)+(-1)*0+0*0, 0*0+(-1)*0+0*1],
    #           [0*0+0*1+1*0, 0*(-1)+0*0+1*0, 0*0+0*0+1*1],
    #           [-1*0+0*1+0*0, -1*(-1)+0*0+0*0, -1*0+0*0+0*1]]
    #        = [[-1, 0, 0],
    #           [0, 0, 1],
    #           [1, 0, 0]]
    # Rx@(Ry@Rz) = [[1,0,0],[0,0,-1],[0,1,0]] @ [[-1,0,0],[0,0,1],[1,0,0]]
    #            = [[-1,0,0],
    #               [0,0,-1]@... let me do element by element
    # Row 0: [1,0,0] @ cols: [-1,0,0] -> -1; [0,0,1] -> 0; [0,0,0] -> 0 => [-1,0,0]
    # Row 1: [0,0,-1] @ cols: [-1,0,0] -> 0; [0,0,1] -> -1; [0,0,0] -> 0 => [0,-1,0]
    # Row 2: [0,1,0] @ cols: [-1,0,0] -> 0; [0,0,1] -> 1; [0,0,0] -> 0 => [0,1,0]
    # So R = [[-1,0,0],[0,-1,0],[0,1,0]]
    # R*p = (-px, -py, pz)
    # Placed point = (-px, -py, pz) + (0,6,7) = (-px, 6-py, pz+7)
    # px in [-4,0] -> -px in [0,4]
    # py in [0,1] -> 6-py in [5,6]
    # pz in [0,1] -> pz+7 in [7,8]
    # Corridor arm 3: x∈[0,1], y∈[5,6], z>=0. So x=-px in [0,4] but corridor only has x∈[0,1].
    # So only points with -px <= 1, i.e., px >= -1, survive. px in [-4,0] so px in [-1,0] -> 40% of x-range.
    # y in [5,6] always ok. z>=0 always ok.
    
    # So at the end pose, only about 1/4 of grid points are in the corridor.
    # This means the maximum possible volume is limited by the end pose.
    
    # The key is to maximize the number of points that survive ALL poses.
    # Points that are in the corridor at both start and end poses, and can be
    # moved through the turns without leaving the corridor.
    
    # Let's design a path that:
    # - Keeps the block centered so that the "survivable" region is as large as possible
    
    # For the yaw turn (0->90), the rotation is about the z-axis.
    # A point p=(px,py,pz) rotated by yaw θ about origin: (px cosθ - py sinθ, px sinθ + py cosθ, pz)
    # Then translated by (x,y,z).
    # For the point to stay in the corridor during the turn, we need the arc to be inside.
    
    # The corridor near the first turn is the region x∈[0,1], y∈[0,1] (the corner).
    # But also arm 1 is x<=0, y∈[0,1], z∈[0,1] and arm 2 is x∈[0,1], y∈[0,5], z∈[0,1].
    
    # The union near the turn is an L-shape in x-y: {x<=0, y∈[0,1]} ∪ {x∈[0,1], y∈[0,5]}
    # which is basically a 1-wide corridor turning from -x to +y.
    
    # For a yaw rotation, the point traces a circular arc in x-y. The bounding box
    # of this arc must be in the corridor. If the rotation center is at (cx,cy) and
    # the point is at distance r, the arc from angle α to β has bounding box
    # [cx + r*cos(max(α,β)), cx + r*cos(min(α,β))] x [cy + r*sin(min), cy + r*sin(max)]
    # (roughly, for small angles).
    
    # To maximize the region that survives, we want the rotation to happen in a
    # region where the corridor is wide enough.
    
    # Let me try a concrete path. I'll use the same general structure as the seed
    # but optimize the key parameters.
    
    # Actually, let me just try to improve on the seed by adjusting the path
    # to keep more points alive. The seed achieves ~1.81 volume.
    
    # Let me try a simpler, more geometrically motivated path.
    
    # Phase 1: Move from (-3,0,0) to (0.5, 0.5, 0.5) with yaw going 0->90
    # The rotation center for yaw should be at (0.5, 0.5) to maximize the
    # space for the arc.
    
    # Actually, the translation (x,y,z) in the pose is the position of the
    # origin of the block's local frame. So when yaw=0, the block is axis-aligned.
    # When yaw=90, the block is rotated 90° about z.
    
    # For a grid point p, at pose (x,y,z,yaw,0,0):
    # placed = (x + px*cos(yaw) - py*sin(yaw), y + px*sin(yaw) + py*cos(yaw), z + pz)
    
    # At yaw=0: placed = (x+px, y+py, z+pz)
    # At yaw=90: placed = (x-px, y+px, z+pz)
    
    # For the point to be in the corridor at yaw=0:
    # x+px <= 0 (arm 1) or (0<=x+px<=1 and 0<=y+py<=5) (arm 2)
    # 0<=y+py<=1 (arm 1) or 0<=y+py<=5 (arm 2)
    # 0<=z+pz<=1
    
    # For the point to be in the corridor at yaw=90:
    # x-px in [0,1] and y+px in [0,5] and z+pz in [0,1] (arm 2)
    # or x-px <= 0 and y+px in [0,1] and z+pz in [0,1] (arm 1)
    
    # This is getting complex. Let me just try to use the seed path and see
    # if I can improve it with a simple optimization.
    
    # Given the complexity, let me return the seed path as a baseline and
    # add a simple local search to improve it.
    
    # Actually, let me just return a well-designed path based on geometric reasoning.
    
    # Key design:
    # 1. The block should move along the centerline of the corridor
    # 2. Turns should happen at the corners with the rotation axis through the center
    # 3. The path should be smooth to minimize the bounding box of rotation arcs
    
    poses = []
    
    # Start: near (-3, 0.5, 0.5) with yaw=0
    # But the evaluator prepends (-3,0,0,0,0,0), so our first pose should be close to that.
    
    # Let's move from (-3,0,0) to the first corner, then turn, then move to second corner, turn, then exit.
    
    # Corner 1 is at approximately (0, 0.5, 0.5) where the corridor turns from -x to +y
    # Corner 2 is at approximately (0.5, 5.5, 0.5) where the corridor turns from +y to +z
    
    # Phase 1: Move from (-3,0,0) to (0, 0.5, 0.5) with yaw 0->45
    n1 = 20
    for i in range(n1 + 1):
        t = i / n1
        x = -3 + 3 * t  # -3 to 0
        y = 0 + 0.5 * t  # 0 to 0.5
        z = 0 + 0.5 * t  # 0 to 0.5
        yaw = 0 + 45 * t  # 0 to 45
        poses.append([x, y, z, yaw, 0, 0])
    
    # Phase 2: Turn yaw from 45 to 90 at the corner, moving slightly into arm 2
    n2 = 10
    for i in range(1, n2 + 1):
        t = i / n2
        x = 0 + 0.5 * t  # 0 to 0.5
        y = 0.5 + 0.5 * t  # 0.5 to 1.0
        z = 0.5
        yaw = 45 + 45 * t  # 45 to 90
        poses.append([x, y, z, yaw, 0, 0])
    
    # Phase 3: Move along arm 2 from y=1 to y=5, yaw=90
    n3 = 20
    for i in range(n3 + 1):
        t = i / n3
        x = 0.5
        y = 1.0 + 4.0 * t  # 1 to 5
        z = 0.5
        yaw = 90
        poses.append([x, y, z, yaw, 0, 0])
    
    # Phase 4: Turn pitch from 0 to 90 at the second corner
    n4 = 10
    for i in range(1, n4 + 1):
        t = i / n4
        x = 0.5
        y = 5.0 + 1.0 * t  # 5 to 6
        z = 0.5 + 0.5 * t  # 0.5 to 1.0
        yaw = 90
        pitch = 0 + 90 * t  # 0 to 90
        poses.append([x, y, z, yaw, pitch, 0])
    
    # Phase 5: Move up in z, pitch=90, yaw=90
    n5 = 20
    for i in range(n5 + 1):
        t = i / n5
        x = 0.5
        y = 6.0
        z = 1.0 + 6.0 * t  # 1 to 7
        yaw = 90
        pitch = 90
        poses.append([x, y, z, yaw, pitch, 0])
    
    # Add roll to match end pose (roll=90)
    # The end pose has roll=90, so we need to add roll somewhere
    # Let's add roll in phase 5
    # Actually, let's add a roll phase
    
    # Remove last pose and add roll
    poses.pop()
    n6 = 5
    for i in range(1, n6 + 1):
        t = i / n6
        x = 0.5
        y = 6.0
        z = 1.0 + 6.0 * (1 - t)  # 7 to 1
        yaw = 90
        pitch = 90
        roll = 0 + 90 * t  # 0 to 90
        poses.append([x, y, z, yaw, pitch, roll])
    
    # Final pose should be near (0, 6, 7, 90, 90, 90)
    # Let's adjust the last few poses
    
    path = np.array(poses, dtype=np.float64)
    
    # Ensure the last pose is close to the end
    path[-1] = [0.0, 6.0, 7.0, 90.0, 90.0, 90.0]
    
    return path
