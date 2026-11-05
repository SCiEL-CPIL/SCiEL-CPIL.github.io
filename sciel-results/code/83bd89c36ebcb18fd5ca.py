"""Optimized snake-in-the-box for Q_9 using bit-parallel exclusion sets."""
import numpy as np

N = 9
NUM_VERTICES = 1 << N  # 512

# Precompute neighbor masks for each vertex
# neighbors_mask[v] is a 512-bit integer with bits set for all vertices adjacent to v
_NEIGHBOR_MASKS = np.zeros(NUM_VERTICES, dtype=np.uint64)
for v in range(NUM_VERTICES):
    mask = 0
    for b in range(N):
        n = v ^ (1 << b)
        mask |= (1 << n)
    _NEIGHBOR_MASKS[v] = mask

def construct():
    """Construct a long induced path (snake) in Q_9.
    
    Returns a list of distinct integers in [0, 512) forming an induced path.
    """
    # Try multiple starting vertices and bit orderings to find a longer snake
    best_path = None
    best_len = 0
    
    # Bit orderings to try: standard, reverse, and some heuristics
    bit_orders = [
        list(range(N)),
        list(range(N - 1, -1, -1)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [0, 8, 1, 7, 2, 6, 3, 5, 4],
        [4, 0, 5, 1, 6, 2, 7, 3, 8],
    ]
    
    for start in range(0, NUM_VERTICES, 32):  # Try every 32nd starting vertex
        for bit_order in bit_orders:
            path = [start]
            used_mask = 1 << start  # 512-bit integer tracking used vertices
            
            while True:
                cur = path[-1]
                cur_neigh_mask = _NEIGHBOR_MASKS[cur]
                
                # Find all neighbors of cur that are not yet used
                available = cur_neigh_mask & ~used_mask
                
                if available == 0:
                    break
                
                # For each candidate neighbor, check if it's adjacent to any non-consecutive path vertex
                # We need to check: for candidate v, is (v's neighbors) & (used_mask without cur) == 0?
                # Actually, we need: no vertex in path[:-1] is adjacent to v.
                # path[:-1] are all used vertices except cur.
                # So we need: neighbor_mask[v] & (used_mask & ~(1 << cur)) == 0
                
                # Iterate through available bits to find a valid extension
                # Get the list of available neighbor vertices
                avail_list = []
                tmp = available
                while tmp:
                    # Get lowest set bit
                    lsb = tmp & (-tmp)
                    idx = lsb.bit_length() - 1
                    avail_list.append(idx)
                    tmp ^= lsb
                
                # Try each available neighbor in the order determined by bit_order
                # We want to try neighbors that differ in bits according to bit_order preference
                # Sort avail_list by the position of the differing bit in bit_order
                def neighbor_sort_key(v):
                    diff = v ^ cur
                    # Find which bit differs
                    bit = diff.bit_length() - 1
                    # Return position in bit_order (lower is preferred)
                    try:
                        return bit_order.index(bit)
                    except ValueError:
                        return N
                
                avail_list.sort(key=neighbor_sort_key)
                
                next_vertex = None
                for v in avail_list:
                    # Check if v is adjacent to any vertex in path[:-1]
                    # i.e., neighbor_mask[v] & (used_mask & ~(1 << cur)) == 0
                    if _NEIGHBOR_MASKS[v] & (used_mask & ~(1 << cur)) == 0:
                        next_vertex = v
                        break
                
                if next_vertex is None:
                    break
                
                path.append(next_vertex)
                used_mask |= (1 << next_vertex)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    if best_path is None:
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check if v is adjacent to any non-consecutive path vertex
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path
    
    return best_path
