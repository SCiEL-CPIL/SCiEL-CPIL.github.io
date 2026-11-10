import numpy as np

def moving_sofa_3d_path() -> np.ndarray:
    # Key poses
    # Start is prepended: (-3, 0, 0, 0, 0, 0)
    # End is appended: (0, 6, 7, 90, 90, 90)
    
    path = [
        # Move to corner 1 (Local origin at 0,0,0)
        [0.0, 0.0, 0.0, 0, 0, 0],
        
        # Rotate in corner 1 (Yaw -90 to align with Y)
        # Use -90 to map (-4,0) to (0,4)
        [0.0, 0.0, 0.0, -90.0, 0, 0],
        
        # Move to corner 2 (Local origin at 0,5,0)
        [0.0, 5.0, 0.0, -90, 0, 0],
        
        # Rotate in corner 2 (Pitch -90 to align with Z)
        # Y to Z is rotation about X. 
        # Ry(pitch) or Rx(roll)?
        # R = Rx(roll) Ry(pitch) Rz(yaw)
        # We want to rotate Y to Z.
        # Ry(-90) maps Y to Z?
        # Ry(theta): x' = x cos + z sin, y' = y, z' = -x sin + z cos
        # No, Ry rotates about Y.
        # Rx(roll) rotates about X.
        # Rx(theta): y' = y cos - z sin, z' = y sin + z cos
        # We have bar along Y. We want it along Z.
        # So we need to rotate Y to Z.
        # Rx(90): y' = -z, z' = y.
        # So Y (0,1,0) -> (0,0,1).
        # So Roll = 90.
        [0.0, 5.0, 0.0, -90, 90, 0],
        
        # Move to end
        # End is (0, 6, 7, 90, 90, 90)
        # We are at (0, 5, 0, -90, 90, 0)
        # We need to reach (0, 6, 7) and orientation (90, 90, 90)
        # Current orientation: Yaw -90, Roll 90.
        # Target: Yaw 90, Pitch 90, Roll 90.
        # This is a complex rotation.
        # Let's just move to the end position and let the evaluator interpolate the rotation?
        # No, we must ensure single axis rotations.
        # Let's move to (0, 6, 7) first?
        # Bar is Z-aligned. Extent z in [0, 4] (local).
        # At t=(0,5,0), z in [0,4].
        # Move to t=(0,6,7).
        # z in [7, 11].
        # Leg 3: z >= 0. Valid.
        [0.0, 6.0, 7.0, -90, 90, 0],
        
        # Now rotate to match end orientation (90, 90, 90)
        # Current: -90, 90, 0
        # Target: 90, 90, 90
        # Change Yaw -90 to 90 (180 deg change? Or -90 to 90 is 180).
        # Change Roll 0 to 90.
        # Let's do Yaw first.
        [0.0, 6.0, 7.0, 90, 90, 0],
        [0.0, 6.0, 7.0, 90, 90, 90]
    ]
    
    return np.array(path, dtype=np.float64)
