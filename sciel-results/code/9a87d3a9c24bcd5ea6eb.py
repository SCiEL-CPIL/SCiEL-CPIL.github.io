import numpy as np

def _verify(comps, N=16):
    # Create all 2^N binary inputs
    masks = np.arange(1 << N, dtype=np.uint8)
    # Expand to bits
    # wires[:, i] is the value of wire i for all masks
    # We can use bit manipulation on the array of masks
    # Actually, creating the 65536x16 array is fine.
    wires = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        wires[:, i] = (masks >> i) & 1
        
    for i, j in comps:
        # Compare and swap
        # min goes to i, max goes to j
        min_val = np.minimum(wires[:, i], wires[:, j])
        max_val = np.maximum(wires[:, i], wires[:, j])
        wires[:, i] = min_val
        wires[:, j] = max_val
        
    # Check sorted
    # A network is sorting if for all masks, wires is non-decreasing
    # i.e., wires[:, k] <= wires[:, k+1] for all k
    diffs = np.diff(wires, axis=1)
    if np.any(diffs < 0):
        return False
    return True
