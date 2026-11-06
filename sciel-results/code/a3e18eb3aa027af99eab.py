import numpy as np

# ... inside construct ...
k4_edge_indices_np = np.array(k4_edge_indices, dtype=np.int32)

def is_valid_fast(col):
    col_arr = np.array(col, dtype=np.int32)
    k4_colors = col_arr[k4_edge_indices_np] # (1820, 6)
    sorted_k4_colors = np.sort(k4_colors, axis=1)
    diffs = np.diff(sorted_k4_colors, axis=1)
    num_duplicates = np.sum(diffs == 0, axis=1)
    # Valid if num_duplicates <= 1 (meaning distinct >= 5)
    return np.all(num_duplicates <= 1)
