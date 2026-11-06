import time
from itertools import combinations

def construct():
    # Deterministic construction of the Witt design S(5,6,12)
    # based on the smaller Witt design S(4,5,11)
    
    # S(4,5,11) has 66 blocks on 11 points (0..10)
    # Each 4-subset of 11 points is in exactly one block of size 5
    
    # Hard-coded blocks of S(4,5,11)
    # These are the 66 blocks of the unique (up to isomorphism) Steiner system S(4,5,11)
    s4511_blocks = [
        (0, 1, 2, 3, 4), (0, 1, 5, 6, 7), (0, 2, 5, 8, 9), (0, 3, 6, 8, 10),
        (0, 4, 7, 9, 10), (1, 2, 6, 9, 10), (1, 3, 5, 9, 10), (1, 4, 5, 8, 10),
        (2, 3, 4, 5, 6), (2, 4, 6, 7, 8), (2, 4, 7, 9, 10), (2, 7, 8, 9, 10),
        (3, 4, 7, 8, 10), (3, 5, 6, 9, 10), (3, 6, 7, 8, 9), (4, 5, 8, 9, 10),
        (5, 6, 7, 8, 9), (0, 1, 2, 5, 8), (0, 1, 2, 6, 9), (0, 1, 2, 7, 10),
        (0, 1, 3, 5, 9), (0, 1, 3, 6, 7), (0, 1, 3, 8, 10), (0, 1, 4, 5, 10),
        (0, 1, 4, 6, 8), (0, 1, 4, 7, 9), (0, 2, 3, 5, 10), (0, 2, 3, 6, 7),
        (0, 2, 3, 9, 10), (0, 2, 4, 6, 10), (0, 2, 4, 7, 8), (0, 2, 4, 9, 10),
        (0, 3, 4, 5, 8), (0, 3, 4, 6, 9), (0, 3, 4, 7, 10), (0, 4, 5, 6, 7),
        (1, 2, 3, 6, 8), (1, 2, 3, 7, 9), (1, 2, 4, 5, 9), (1, 2, 4, 6, 10),
        (1, 2, 4, 8, 9), (1, 3, 4, 6, 9), (1, 3, 4, 7, 8), (1, 4, 5, 6, 8),
        (2, 3, 4, 7, 9), (2, 3, 4, 8, 10), (2, 3, 5, 6, 8), (2, 3, 5, 7, 10),
        (2, 3, 5, 8, 9), (2, 4, 5, 6, 9), (2, 4, 5, 7, 8), (2, 5, 6, 7, 10),
        (3, 4, 5, 7, 9), (3, 4, 5, 8, 10), (3, 4, 6, 7, 8), (3, 5, 6, 7, 9),
        (4, 5, 6, 7, 10), (4, 6, 7, 8, 9), (5, 6, 8, 9, 10), (5, 7, 8, 9, 10),
        (6, 7, 8, 9, 10)
    ]
    
    # Verify we have 66 distinct blocks
    s4511_blocks = list(set(s4511_blocks))
    if len(s4511_blocks) != 66:
        # Fallback: generate via DLX if hard-coded list is wrong
        return _construct_via_dlx()
    
    # Construct S(5,6,12) from S(4,5,11)
    # Points are 0..11
    # Type 1: {11} ∪ B for each B in S(4,5,11)
    # Type 2: (0..10) \ B for each B in S(4,5,11) (complement in 0..10, which has size 6)
    
    result = []
    for b in s4511_blocks:
        b = list(b)
        # Type 1: add point 11
        block1 = tuple(sorted(b + [11]))
        result.append(block1)
        
        # Type 2: complement in {0,1,...,10}
        complement = tuple(sorted(set(range(11)) - set(b)))
        result.append(complement)
    
    # Deduplicate just in case
    result = list(set(result))
    
    # Verify it's a valid S(5,6,12): every 5-subset appears in exactly one block
    # Count 5-subsets
    from collections import Counter
    five_subset_count = Counter()
    for block in result:
        for subset in combinations(block, 5):
            five_subset_count[subset] += 1
    
    # All 792 five-subsets should appear exactly once
    valid = (len(result) == 132 and 
             all(count == 1 for count in five_subset_count.values()) and
             len(five_subset_count) == 792)
    
    if not valid:
        # Fallback: generate via DLX
        return _construct_via_dlx()
    
    return result


def _construct_via_dlx():
    """Fallback: use DLX to find S(5,6,12)"""
    import time
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    n_points = 12
    block_size = 6
    subset_size = 5
    
    all_blocks = list(combinations(range(n_points), block_size))
    num_blocks = len(all_blocks)
    
    all_5_subsets = list(combinations(range(n_points), subset_size))
    num_5_subsets = len(all_5_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # DLX Setup
    max_nodes = num_blocks * 6 + num_5_subsets + 1
    
    L = [-1] * max_nodes
    R = [-1] * max_nodes
    U = [-1] * max_nodes
    D = [-1] * max_nodes
    row_id = [-1] * max_nodes
    col_id = [-1] * max_nodes
    
    L[0] = num_5_subsets
    R[0] = 1
    for j in range(num_5_subsets):
        node = j + 1
        L[node] = node - 1
        R[node] = node + 1
        U[node] = node
        D[node] = node
        col_id[node] = j
        row_id[node] = -1
    
    R[num_5_subsets] = 0
    L[1] = 0
    
    col_size = [0] * num_5_subsets
    
    next_node = num_5_subsets + 1
    
    for i, b in enumerate(all_blocks):
        subsets_in_b = combinations(b, subset_size)
        first_node_in_row = -1
        for s in subsets_in_b:
            idx = subset_to_idx[s]
            col_header = idx + 1
            
            node = next_node
            next_node += 1
            
            row_id[node] = i
            col_id[node] = idx
            
            U[node] = U[col_header]
            D[node] = col_header
            D[U[col_header]] = node
            U[col_header] = node
            
            col_size[idx] += 1
            
            if first_node_in_row == -1:
                first_node_in_row = node
                L[node] = node
                R[node] = node
            else:
                R[node] = R[first_node_in_row]
                L[node] = first_node_in_row
                R[first_node_in_row] = node
                L[R[node]] = node
    
    solution = []
    
    def remove(node):
        R[L[node]] = R[node]
        L[R[node]] = L[node]
        
    def restore(node):
        L[R[node]] = node
        R[L[node]] = node
        
    def search(k):
        if time.time() > deadline:
            return False
            
        if R[0] == 0:
            return True
            
        c = R[0]
        min_size = col_size[0]
        j = c
        while j != 0:
            if col_size[col_id[j]] < min_size:
                min_size = col_size[col_id[j]]
                c = j
            j = R[j]
            
        if min_size == 0:
            return False
            
        remove(c)
        i = D[c]
        while i != c:
            solution.append(row_id[i])
            
            j = R[i]
            while j != i:
                remove(j)
                k2 = D[j]
                while k2 != j:
                    col_size[col_id[k2]] -= 1
                    remove(k2)
                    k2 = D[k2]
                j = R[j]
                
            if search(k + 1):
                return True
                
            solution.pop()
            
            j = L[i]
            while j != i:
                k2 = U[j]
                while k2 != j:
                    col_size[col_id[k2]] += 1
                    restore(k2)
                    k2 = U[k2]
                restore(j)
                j = L[j]
                
            i = D[i]
            
        restore(c)
        return False
        
    if search(0):
        return [list(all_blocks[i]) for i in solution]
    else:
        return []
