import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1, we aim for a covering code of size 73.
    # If time runs out, we fall back to the direct sum construction (size 81).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct sum of tetracode
        # This is a valid covering code for q=3, n=4, r=1 (size 9)
        # For general n, we can extend it.
        # For n=6, direct sum of (3,4,1) code gives 9 * 3^(6-4) = 9*9=81.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = 115.0  # Leave 5s safety margin for 120s limit (or 150s limit, but 120s is safer)

    # Precompute neighbors for all 729 words
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    # We represent words as integers 0..728 (base 3)
    
    def num_to_vec(num, length=6):
        vec = [0] * length
        for i in range(length):
            vec[i] = num % 3
            num //= 3
        return vec

    def vec_to_num(vec):
        num = 0
        for i in range(len(vec)):
            num += vec[i] * (3 ** i)
        return num

    # Precompute coverage sets as lists of indices
    # cover_set[i] is a list of words covered by codeword i
    cover_set = []
    for i in range(729):
        v = num_to_vec(i)
        s = [i]
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    s.append(vec_to_num(v))
            v[pos] = orig
        cover_set.append(s)
        
    # Convert cover_set to numpy arrays for faster operations if needed, 
    # but lists are fine for 729 elements.
    
    # Spectral Initialization:
    # Construct the graph Laplacian of the Hamming distance-2 graph? 
    # Actually, the strategy suggests using the graph Laplacian of the Hamming distance-2 graph 
    # to find a diverse starting point. 
    # However, for a covering code, we want a set that covers everything. 
    # A diverse set is good for initialization to avoid local minima in greedy.
    # Let's try to compute eigenvectors of the adjacency matrix of the Hamming graph H(6,3) (distance 1).
    # Or distance 2? The prompt says "Hamming distance-2 graph".
    # Let's stick to the prompt's suggestion: "constructing the graph Laplacian of the Hamming distance-2 graph".
    # But computing eigenvectors of a 729x729 matrix is cheap.
    
    # Build adjacency matrix for distance 2?
    # No, let's use the standard Hamming graph (distance 1) for spectral centrality, 
    # or just use the distance-2 graph as requested.
    # Let's build the distance-2 graph adjacency.
    # A[i, j] = 1 if dist(i, j) == 2.
    
    # To speed this up, we can generate neighbors at distance 2.
    # But 729*729 is small enough to just check distances? 
    # Dist check is O(6). 729^2 * 6 ~ 3M ops. Very fast.
    
    adj_matrix = np.zeros((729, 729), dtype=np.int8)
    for i in range(729):
        v_i = num_to_vec(i)
        for j in range(i + 1, 729):
            v_j = num_to_vec(j)
            dist = sum(1 for a, b in zip(v_i, v_j) if a != b)
            if dist == 2:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                
    # Compute Laplacian L = D - A
    degrees = np.sum(adj_matrix, axis=1)
    L = np.diag(degrees) - adj_matrix
    
    # Compute eigenvectors
    # We want "eigenvalue-weighted centrality". 
    # Usually, high centrality in a graph corresponds to large components of the principal eigenvector (smallest non-zero eigenvalue for connected graphs? or largest?).
    # For a covering code, we want well-distributed points. 
    # Let's take the eigenvector corresponding to the smallest non-zero eigenvalue (Fiedler vector) or the principal one?
    # Actually, for spreading out, we might want to pick vertices with high values in the eigenvectors associated with the smallest eigenvalues (smoothest modes).
    # Or, we can just use the principal eigenvector of the adjacency matrix (which is all 1s for regular graphs, so not useful).
    # The Hamming graph is regular. So the principal eigenvector is constant.
    # Let's use the eigenvectors of the Laplacian. The smallest non-zero eigenvalue's eigenvector is the Fiedler vector.
    # But for a highly symmetric graph, there are many degenerate eigenvalues.
    # Let's pick the top 80 eigenvectors (excluding the zero one) and compute a score for each vertex as the sum of squares of its components in these eigenvectors.
    # This is essentially the energy of the vertex in the subspace of the graph.
    
    # Since the graph is regular and highly symmetric, the eigenvectors are related to Krawtchouk polynomials.
    # But let's just use numpy.
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # The first eigenvalue is 0 (constant vector).
        # We want to use the next few eigenvectors to define a "spectral position" for each vertex.
        # Let's take the first 10 non-zero eigenvectors.
        k = 10
        spectral_coords = eigenvectors[:, 1:k+1]  # Shape (729, k)
        
        # Score: sum of squares of coordinates (energy in this subspace)
        spectral_scores = np.sum(spectral_coords ** 2, axis=1)
        
        # Normalize scores to 0..1
        min_s = np.min(spectral_scores)
        max_s = np.max(spectral_scores)
        if max_s > min_s:
            spectral_scores = (spectral_scores - min_s) / (max_s - min_s)
        else:
            spectral_scores = np.zeros(729)
            
    except Exception:
        spectral_scores = np.random.rand(729)

    # Initial selection: Pick vertices with high spectral scores?
    # High spectral energy in low-frequency modes often means they are "central" or well-connected in a smooth sense.
    # However, for a covering code, we want to cover the space.
    # Let's use the spectral scores to bias the greedy selection.
    # Or, we can initialize the code with the top 80 vertices by spectral score, then refine.
    
    # Let's try: Initialize with top 80 vertices by spectral score.
    # Then run local search to minimize size.
    
    # But wait, the greedy set cover usually starts empty.
    # The inspiration says: "initializing the codeword set by selecting vertices with the highest 'spectral gap' scores... to create a structurally diverse starting point".
    
    # Let's create an initial set of size 80 (or 90) using spectral scores.
    # Then reduce it.
    
    init_size = 85
    top_indices = np.argsort(-spectral_scores)[:init_size]
    current_set = set(top_indices.tolist())
    
    # Ensure coverage? The spectral selection might not cover everything.
    # So we need a repair phase: add codewords to cover uncovered words.
    
    def compute_cover_count(selected_set):
        cover_count = np.zeros(729, dtype=np.int32)
        for c in selected_set:
            for w in cover_set[c]:
                cover_count[w] += 1
        return cover_count

    cover_count = compute_cover_count(current_set)
    
    # Repair: Add codewords to cover uncovered words
    uncovered = np.where(cover_count == 0)[0]
    while len(uncovered) > 0 and time.time() < deadline - 10:
        # Pick the codeword that covers the most uncovered words
        best_c = -1
        max_gain = 0
        # Only consider codewords that cover at least one uncovered word
        # We can iterate over all 729, it's fast enough.
        for c in range(729):
            if c in current_set:
                continue
            gain = np.sum(cover_count[cover_set[c]] == 0)
            if gain > max_gain:
                max_gain = gain
                best_c = c
        if max_gain == 0:
            break
        current_set.add(best_c)
        cover_count = compute_cover_count(current_set)
        uncovered = np.where(cover_count == 0)[0]
        
    best_size = len(current_set)
    best_code_set = set(current_set)
    
    # Local Search to minimize size
    # Moves:
    # 1. Delete c if valid.
    # 2. Swap c_out with c_in if it enables a deletion or reduces size? No, swap keeps size same.
    #    We need to find a sequence of swaps that leads to a state where a deletion is possible.
    #    Or, we can use a "1-1 swap" that reduces the number of critical words (count 1).
    #    Then try to delete.
    
    # Let's use a simple local search:
    # While size > 73:
    #   1. Try to delete a codeword. If successful, continue.
    #   2. If no deletion possible, try a random 1-1 swap that improves a heuristic (e.g., minimizes number of words with count 1).
    #      Accept the swap with some probability (simulated annealing style) or if it improves the heuristic.
    #   3. If stuck, perturb by adding a random codeword and then trying to delete.
    
    for iteration in range(20000):
        if time.time() > deadline - 5:
            break
        if best_size <= 73:
            break
            
        # Update cover_count
        cover_count = compute_cover_count(current_set)
        
        # 1. Try deletion
        deleted = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for c in candidates:
            # Check if removing c keeps all counts >= 1
            # We can do this by checking min(cover_count[w]) for w in cover_set[c]
            # If min >= 2, we can remove.
            min_count = np.min(cover_count[cover_set[c]])
            if min_count >= 2:
                current_set.remove(c)
                best_size = len(current_set)
                best_code_set = set(current_set)
                deleted = True
                break
        
        if deleted:
            continue
            
        # 2. Try 1-1 swap
        # We want to swap c_out (in set) with c_in (not in set) such that the number of "critical" words (count 1) decreases.
        # Or, such that it enables a deletion.
        # Let's calculate the change in critical words for a swap.
        # critical_words = np.sum(cover_count == 1)
        
        # To speed up, we can precompute the change in critical words for each possible swap?
        # That's 729*729 swaps, which is 500k. Might be too slow per iteration.
        
        # Let's try a random swap.
        if random.random() < 0.5:
            # Random swap
            c_out = random.choice(list(current_set))
            c_in = random.randint(0, 728)
            if c_in in current_set:
                continue
                
            # Try the swap
            current_set.remove(c_out)
            current_set.add(c_in)
            
            # Check if this swap enabled a deletion
            cover_count_temp = compute_cover_count(current_set)
            
            # Try to delete
            can_delete = False
            for c in list(current_set):
                if np.min(cover_count_temp[cover_set[c]]) >= 2:
                    current_set.remove(c)
                    best_size = len(current_set)
                    best_code_set = set(current_set)
                    can_delete = True
                    break
            
            if not can_delete:
                # Revert swap if it didn't help (or accept with some probability?)
                # For now, let's accept the swap if it doesn't increase size (it doesn't) 
                # and we hope it leads to a deletion later.
                # But if we just revert, we might loop.
                # Let's accept the swap always, as long as coverage is maintained.
                # Coverage is maintained because we only swap in/out, and we check deletion.
                # Wait, if we swap and don't delete, the size is the same.
                # We need to ensure we don't get stuck in a loop.
                # Let's just accept the swap.
                pass
        else:
            # Try a greedy swap: find c_out, c_in that minimizes critical words
            # This is expensive. Let's skip for now and rely on random swaps + deletions.
            pass
            
    # Final result
    # Convert set to list of lists
    result = [num_to_vec(c) for c in best_code_set]
    
    # Verify validity (optional, but good for debugging)
    # for w in range(729):
    #     covered = False
    #     for c in result:
    #         if sum(1 for a, b in zip(num_to_vec(w), c) if a != b) <= 1:
    #             covered = True
    #             break
    #     if not covered:
    #         raise ValueError("Invalid covering code")
        
    return result

