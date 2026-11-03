import random
import time

def construct(n):
    if n == 1:
        return "1"
    
    # Generate S_{n-1}
    S_prev = construct(n-1)
    
    # Extract unique permutations of 1..n-1 in order of appearance
    perms = []
    seen = set()
    m = n - 1
    for i in range(len(S_prev) - m + 1):
        w = tuple(int(c) for c in S_prev[i:i+m])
        if w not in seen:
            seen.add(w)
            perms.append(w)
            
    # Optimize the order of permutations to maximize total overlap
    # Overlap between P_i and P_{i+1} is the length of the longest suffix of P_i 
    # that is a prefix of P_{i+1}.
    # We want to maximize sum of overlaps.
    
    M = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = overlap(perms[i], perms[j])
    overlap_matrix = [[0]*M for _ in range(M)]
    for i in range(M):
        pi = perms[i]
        for j in range(M):
            pj = perms[j]
            max_k = 0
            # Check k from 1 to m
            # If pi[-k:] == pj[:k], update max_k
            # We can break if mismatch? 
            # If pi[-k:] != pj[:k], then for k+1, pi[-(k+1):] ends with pi[-k:], 
            # and pj[:k+1] starts with pj[:k]. 
            # So if the k part doesn't match, the k+1 part can't match?
            # Wait.
            # pi = (1, 2, 3). pj = (2, 3, 4).
            # k=1: pi[-1:]=(3), pj[:1]=(2). No.
            # k=2: pi[-2:]=(2,3), pj[:2]=(2,3). Yes.
            # So we cannot break on mismatch at k=1. We must check all k.
            for k in range(1, m+1):
                if pi[-k:] == pj[:k]:
                    max_k = k
            overlap_matrix[i][j] = max_k
            
    # Current order is the standard one (0, 1, ..., M-1)
    order = list(range(M))
    
    def get_score(order):
        score = 0
        for i in range(M-1):
            score += overlap_matrix[order[i]][order[i+1]]
        return score
        
    current_score = get_score(order)
    
    # Local Search: Simulated Annealing
    # Time limit: ~60 seconds for this part
    start_time = time.time()
    timeout = 60
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while T > T_min and time.time() - start_time < timeout:
        # Pick two random indices to swap
        i = random.randint(0, M-1)
        j = random.randint(0, M-1)
        if i == j:
            continue
            
        # Calculate delta score
        # Remove edges (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # Add edges (i-1, j), (j, i+1), (j-1, i), (i, j+1)
        
        old_score = 0
        new_score = 0
        
        # Helper to get neighbor index safely
        def get_neighbor(idx, pos):
            if pos < 0: return -1
            if pos >= M: return -1
            return order[pos]
            
        # Edges involving i
        i_prev = get_neighbor(i, i-1)
        i_next = get_neighbor(i, i+1)
        
        # Edges involving j
        j_prev = get_neighbor(j, j-1)
        j_next = get_neighbor(j, j+1)
        
        # If i and j are adjacent, the calculation is simpler
        if abs(i-j) == 1:
            # Case i < j (j = i+1)
            if i < j:
                # Old: (i-1, i), (i, i+1), (i+1, i+2)
                # New: (i-1, i+1), (i+1, i), (i, i+2)
                if i_prev != -1:
                    old_score += overlap_matrix[order[i_prev]][order[i]]
                    new_score += overlap_matrix[order[i_prev]][order[j]]
                
                old_score += overlap_matrix[order[i]][order[j]]
                new_score += overlap_matrix[order[j]][order[i]]
                
                if i_next != -1:
                    old_score += overlap_matrix[order[j]][order[i_next]]
                    new_score += overlap_matrix[order[i]][order[i_next]]
            else:
                # Case j < i (i = j+1)
                if j_prev != -1:
                    old_score += overlap_matrix[order[j_prev]][order[j]]
                    new_score += overlap_matrix[order[j_prev]][order[i]]
                
                old_score += overlap_matrix[order[j]][order[i]]
                new_score += overlap_matrix[order[i]][order[j]]
                
                if i_next != -1:
                    old_score += overlap_matrix[order[i]][order[i_next]]
                    new_score += overlap_matrix[order[j]][order[i_next]]
        else:
            # Non-adjacent
            if i_prev != -1:
                old_score += overlap_matrix[order[i_prev]][order[i]]
                new_score += overlap_matrix[order[i_prev]][order[j]]
            if i_next != -1:
                old_score += overlap_matrix[order[i]][order[i_next]]
                new_score += overlap_matrix[order[j]][order[i_next]]
                
            if j_prev != -1:
                old_score += overlap_matrix[order[j_prev]][order[j]]
                new_score += overlap_matrix[order[j_prev]][order[i]]
            if j_next != -1:
                old_score += overlap_matrix[order[j]][order[j_next]]
                new_score += overlap_matrix[order[i]][order[j_next]]
                
        delta = new_score - old_score
        
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            order[i], order[j] = order[j], order[i]
            current_score += delta
            
        T *= cooling_rate

    # Reconstruct string
    word = ""
    for idx in order:
        p = perms[idx]
        block = "".join(map(str, p)) + str(n) + "".join(map(str, p))
        if not word:
            word = block
        else:
            # Find overlap
            # We can use the precomputed overlap for the last perm in word?
            # No, the suffix of word is not just the last perm.
            # But we know the suffix of the PREVIOUS BLOCK is the previous perm.
            # And the current word ends with the previous block.
            # So the suffix of word of length <= m is the suffix of the previous perm.
            # So we can just check overlap between previous perm and current perm.
            # Wait, is it possible that the overlap extends into the previous block?
            # Suffix of word is ... P_{prev}.
            # Prefix of block is P_{curr}.
            # Max overlap is m.
            # So we just need overlap_matrix[prev_idx][curr_idx].
            prev_idx = order[order.index(idx)-1] # This is slow
            # Better to keep track
            pass
            
    # Let's do the reconstruction properly
    word = ""
    prev_idx = -1
    for k, idx in enumerate(order):
        p = perms[idx]
        block = "".join(map(str, p)) + str(n) + "".join(map(str, p))
        if k == 0:
            word = block
        else:
            ov = overlap_matrix[prev_idx][idx]
            word += block[ov:]
        prev_idx = idx
        
    return word
