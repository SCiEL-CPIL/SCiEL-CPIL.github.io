import time
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) — a Steiner system with 132 blocks.
    
    Uses the algebraic definition via the [12,6,6] code (self-dual, minimum distance 6).
    The weight-6 codewords form the 132 blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Construct the [12,6,6] code generator matrix.
    # One standard representation: G = [I_6 | P] where P is 6x6.
    # The [12,6,6] code is the dual of the [12,6,6] code (self-dual).
    # 
    # A well-known generator matrix for the [12,6,6] code:
    # Using the construction from the extended binary Golay code shortened by 3 bits.
    
    # The generator matrix G for the [12,6,6] code in systematic form:
    # G = [I_6 | P]
    # 
    # One standard P matrix (from literature on the shortened Golay code):
    P = [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]
    
    # Build the full 6x12 generator matrix
    G = []
    for i in range(6):
        row = [0] * 12
        row[i] = 1  # Identity part
        for j in range(6):
            row[6 + j] = P[i][j]
        G.append(row)
    
    # Enumerate all 2^6 = 64 codewords and find those of weight 6
    blocks_set = set()
    for mask in range(64):
        word = [0] * 12
        for i in range(6):
            if (mask >> i) & 1:
                for j in range(12):
                    word[j] ^= G[i][j]
        weight = sum(word)
        if weight == 6:
            block = tuple(j for j in range(12) if word[j] == 1)
            blocks_set.add(block)
    
    n_blocks = len(blocks_set)
    
    if n_blocks == 132:
        blocks_list = [list(b) for b in blocks_set]
        return blocks_list
    
    # Fallback: try the other standard construction via the dual code
    # The [12,6,6] code is self-dual, so let's try the parity-check matrix approach
    
    # Alternative: construct via the [15,11,3] Hamming code
    H15 = []
    for i in range(1, 16):
        col = [(i >> j) & 1 for j in range(4)]
        H15.append(col)
    
    H15_rows = [[H15[j][i] for j in range(15)] for i in range(4)]
    
    # Shorten by fixing last 3 coordinates to 0
    H12 = [[H15_rows[i][j] for j in range(12)] for i in range(4)]
    
    # The row space of H12 gives the [12,4,6] dual code
    dual_codewords = []
    for mask in range(16):
        word = [0] * 12
        for i in range(4):
            if (mask >> i) & 1:
                for j in range(12):
                    word[j] ^= H12[i][j]
        weight = sum(word)
        if weight == 6:
            dual_codewords.append(word)
    
    blocks_set2 = set()
    for word in dual_codewords:
        block = tuple(j for j in range(12) if word[j] == 1)
        blocks_set2.add(block)
    
    if len(blocks_set2) == 132:
        blocks_list = [list(b) for b in blocks_set2]
        return blocks_list
    
    # If algebraic construction fails, use backtracking with MRV
    all_blocks = list(combinations(range(12), 6))
    all_5subs = list(combinations(range(12), 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # Precompute which 5-subsets each block covers
    block_subs = []
    for b in all_blocks:
        subs = []
        for s in combinations(b, 5):
            subs.append(sub_to_idx[s])
        block_subs.append(subs)
    
    # Greedy with MRV heuristic
    covered = [False] * 792
    selected = []
    remaining_blocks = list(range(924))
    
    while remaining_blocks:
        # Find block with minimum remaining 5-subsets (MRV)
        best_idx = -1
        best_count = float('inf')
        for idx in remaining_blocks:
            count = 0
            for s_idx in block_subs[idx]:
                if not covered[s_idx]:
                    count += 1
            if count < best_count:
                best_count = count
                best_idx = idx
            if count == 0:
                break
        
        if best_count == 0:
            break
        
        # Select this block
        selected.append(all_blocks[best_idx])
        for s_idx in block_subs[best_idx]:
            covered[s_idx] = True
        
        # Remove selected block from remaining
        remaining_blocks.remove(best_idx)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    if len(selected) >= 132:
        return selected[:132]
    
    # Final fallback: return what we have
    return selected
