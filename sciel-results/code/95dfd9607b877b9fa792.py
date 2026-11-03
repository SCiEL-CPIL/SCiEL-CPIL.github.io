import numpy as np
import time
import itertools
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected in this task, but good practice)
        # Simple greedy
        return greedy_cover(q, n, radius, deadline)

    # 1. Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words) # 6561
    
    # 2. Generate Spectral Embedding
    # Basis for K3
    # e0 = [1, 1, 1]
    # e1 = [1, -1, 0]
    # e2 = [1, 0, -1]
    
    # We construct the 16 eigenvectors corresponding to eigenvalue 3.
    # These are formed by placing e1 or e2 in one of the 8 positions, and e0 elsewhere.
    # Since e0 is constant, the value of the eigenvector at a word w is:
    # v(w) = (val_{pos}(w[pos])) * (product of 1s) = val_{pos}(w[pos])
    # Wait, the tensor product of vectors.
    # If v = e1 (pos 0) \otimes e0 (pos 1) ...
    # v(w) = e1(w[0]) * e0(w[1]) * ... = (1 if w[0]==0, -1 if w[0]==1, 0 if w[0]==2) * 1 * ...
    # So the 16 eigenvectors simply correspond to the "signed" values of the coordinates?
    # Let's check.
    # e1 = [1, -1, 0]. e2 = [1, 0, -1].
    # For pos i, we have two vectors.
    # Vector i_1: w[i]==0 -> 1, w[i]==1 -> -1, w[i]==2 -> 0.
    # Vector i_2: w[i]==0 -> 1, w[i]==1 -> 0, w[i]==2 -> -1.
    
    # Let's build the embedding matrix X of shape (6561, 16)
    X = np.zeros((N, 16), dtype=np.float32)
    for i in range(n):
        # Vector i_1
        col1 = np.where(all_words[:, i] == 0, 1.0, np.where(all_words[:, i] == 1, -1.0, 0.0))
        # Vector i_2
        col2 = np.where(all_words[:, i] == 0, 1.0, np.where(all_words[:, i] == 2, -1.0, 0.0))
        X[:, 2*i] = col1
        X[:, 2*i+1] = col2
        
    # 3. Cluster using K-Means
    # Target size approx 500-600
    target_k = 550
    # Use kmeans2 from scipy
    # Note: kmeans2 can be slow if K is large, but 550 is okay.
    # We need to ensure we don't hit the deadline here.
    if time.time() > deadline - 20:
        centroids, labels = kmeans2(X, target_k, minit='++', seed=42)
    else:
        centroids, labels = kmeans2(X, target_k, minit='++', seed=42)
        
    # 4. Map centroids to nearest words
    # For each centroid, find the word in all_words closest to it in X-space?
    # Or find the word that minimizes distance to centroid in original space?
    # The embedding is linear-ish. Nearest in X-space is a good proxy.
    # But we need valid codewords.
    # We can just pick the word corresponding to the label with the most members?
    # Or pick the word closest to the centroid in X-space.
    
    initial_codewords = []
    for c in centroids:
        # Find index of word closest to centroid in X-space
        dists = np.sum((X - c)**2, axis=1)
        idx = np.argmin(dists)
        initial_codewords.append(all_words[idx])
        
    # Convert to list of lists
    initial_code = [list(w) for w in initial_codewords]
    
    # 5. Refine with ILS
    best_code = initial_code
    best_size = len(best_code)
    
    # Precompute neighborhoods for fast coverage check
    # neighbor_map: word_index -> list of indices within distance 1
    # This is static.
    # But we operate on the set of codewords.
    
    # Optimization: Use the cover_count array approach.
    
    def get_cover_count(code_indices):
        """
        code_indices: list of indices into all_words
        Returns: np.array of size N with coverage counts
        """
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_indices:
            # Get neighbors of all_words[idx]
            # Neighbors are at Hamming distance <= 1
            # We can precompute the list of neighbor indices for each word.
            pass
        return counts

    # Precompute neighbors for all 6561 words
    # neighbors[i] = list of j such that d(i, j) <= 1
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                # Create new word
                new_w = w.copy()
                new_w[pos] = val
                # Find index of new_w
                # Since all_words is sorted (lexicographic product), index is base-3 representation
                # new_w is a tuple of ints.
                # idx = sum(new_w[pos] * 3**(n-1-pos))
                idx = 0
                for k in range(n):
                    idx = idx * 3 + new_w[k]
                neigh.append(idx)
        neighbors.append(neigh)
        
    # Function to check if code (list of indices) covers all
    def is_valid(code_indices):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_indices:
            for n_idx in neighbors[idx]:
                counts[n_idx] += 1
        return np.all(counts > 0)

    # Function to get redundant indices
    def find_redundant(code_indices):
        redundant = []
        for idx in code_indices:
            # Temporarily remove
            counts = np.zeros(N, dtype=np.int32)
            for other in code_indices:
                if other == idx: continue
                for n_idx in neighbors[other]:
                    counts[n_idx] += 1
            if np.all(counts > 0):
                redundant.append(idx)
        return redundant

    # ILS Loop
    current_code = [list(w) for w in initial_codewords]
    # Convert to indices for fast processing
    # Map word -> index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    current_indices = [word_to_idx[tuple(w)] for w in current_codewords]
    
    # Remove duplicates in current_indices (K-Means might pick same word for multiple centroids)
    current_indices = list(set(current_indices))
    
    # Initial Reduction
    changed = True
    while changed:
        changed = False
        for idx in current_indices[:]:
            if is_valid([i for i in current_indices if i != idx]):
                current_indices.remove(idx)
                changed = True
                
    best_indices = current_indices[:]
    best_size = len(best_indices)
    
    # ILS with Perturbation
    temp = 1.0
    for i in range(5000): # Max iterations
        if time.time() > deadline - 5:
            break
            
        # Perturbation: Remove k random words, then greedily restore
        k = max(1, len(current_indices) // 20)
        perturbed = current_indices[:]
        if len(perturbed) > k:
            to_remove = np.random.choice(len(perturbed), k, replace=False)
            for idx in to_remove:
                perturbed.pop(idx)
                
        # Greedy Restore
        # Find uncovered words
        counts = np.zeros(N, dtype=np.int32)
        for idx in perturbed:
            for n_idx in neighbors[idx]:
                counts[n_idx] += 1
        uncovered = np.where(counts == 0)[0]
        
        while len(uncovered) > 0:
            if time.time() > deadline - 5:
                break
            # Find word covering most uncovered
            best_idx = -1
            max_cov = -1
            # Sample a subset of candidate words if N is large? N=6561 is small.
            # Check all words? 6561 * 17 ops = 100k ops. Fast.
            for cand in range(N):
                if cand in perturbed: continue
                cov = 0
                for n_idx in neighbors[cand]:
                    if counts[n_idx] == 0:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_idx = cand
            if best_idx == -1: break
            perturbed.append(best_idx)
            # Update counts
            for n_idx in neighbors[best_idx]:
                counts[n_idx] += 1
            # Update uncovered
            uncovered = np.where(counts == 0)[0]
            
        # Reduction
        changed = True
        while changed:
            changed = False
            for idx in perturbed[:]:
                if is_valid([i for i in perturbed if i != idx]):
                    perturbed.remove(idx)
                    changed = True
                    
        # Acceptance
        if len(perturbed) < best_size:
            best_indices = perturbed[:]
            best_size = len(best_indices)
            current_indices = perturbed[:]
            temp = max(0.1, temp * 0.9)
        else:
            # Accept with probability
            if np.random.rand() < np.exp(-(len(perturbed) - best_size) / temp):
                current_indices = perturbed[:]
            temp = max(0.1, temp * 0.95)
            
    # Convert best_indices back to list of lists
    return [list(all_words[i]) for i in best_indices]

def greedy_cover(q, n, radius, deadline):
    # Simple fallback
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = w.copy()
                new_w[pos] = val
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                neigh.append(idx)
        neighbors.append(neigh)
        
    covered = np.zeros(N, dtype=bool)
    code = []
    while not np.all(covered):
        if time.time() > deadline: break
        best_idx = -1
        max_new = -1
        for i in range(N):
            new_cov = 0
            for n_idx in neighbors[i]:
                if not covered[n_idx]:
                    new_cov += 1
            if new_cov > max_new:
                max_new = new_cov
                best_idx = i
        if best_idx == -1: break
        code.append(best_idx)
        for n_idx in neighbors[best_idx]:
            covered[n_idx] = True
            
    return [list(all_words[i]) for i in code]
