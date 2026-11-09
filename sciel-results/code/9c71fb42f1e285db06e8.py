"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    points = []
    
    # Add all points with hex distance <= 2
    # Hex distance for (a, b) is max(|a|, |b|, |a+b|) is incorrect.
    # Cube distance is max(|x|, |y|, |z|) where x=a, z=b, y=-a-b.
    # So dist = max(|a|, |b|, |a+b|).
    # Wait, max(|a|, |b|, |a+b|) is not the same as (|a|+|b|+|a+b|)/2?
    # Let's check (1,1). max(1,1,2)=2. (1+1+2)/2 = 2.
    # Let's check (2,0). max(2,0,2)=2. (2+0+2)/2 = 2.
    # Let's check (1,-1). max(1,1,0)=1. (1+1+0)/2 = 1.
    # Yes, dist = max(|a|, |b|, |a+b|).
    
    for a in range(-2, 3):
        for b in range(-2, 3):
            d = max(abs(a), abs(b), abs(a+b))
            if d <= 2:
                points.append([a, b])
                
    # Add points with hex distance == 3
    # Vertices of ring 3
    vertices = [(3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3)]
    vertex_set = set(vertices)
    
    added_vertex = False
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = max(abs(a), abs(b), abs(a+b))
            if d == 3:
                if (a, b) in vertex_set:
                    # Add only one vertex
                    if not added_vertex:
                        points.append([a, b])
                        added_vertex = True
                else:
                    points.append([a, b])
                    
    return points
# EVOLVE-BLOCK-END
