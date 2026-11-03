"""Covering code K_3(6,1) - spectral partitioning with KL refinement."""
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    total = q ** n  # 729 for q=3, n=6
    
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return w
    
    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx
    
    # Precompute balls for all words
    balls = [None] * total
    for w_idx in range(total):
        w = idx_to_word(w_idx)
        ball = [w_idx]
        for pos in range(n):
            for delta in range(1, q):
                new_w = w[:]
                new_w[pos] = (w[pos] + delta) % q
                ball.append(word_to_idx(new_w))
        balls[w_idx] = set(ball)
    
    # Build adjacency matrix of Hamming graph
    # Two vertices are adjacent if they differ in exactly one position
    adj = np.zeros((total, total), dtype=np.int8)
    for w_idx in range(total):
        w = idx_to_word(w_idx)
        for pos in range(n):
            for delta in range(1, q):
                new_w = w[:]
                new_w[pos] = (w[pos] + delta) % q
                neighbor = word_to_idx(new_w)
                adj[w_idx, neighbor] = 1
                adj[neighbor, w_idx] = 1
    
    # Compute Laplacian
    L = laplacian(csr_matrix(adj))
    
    # Get Fiedler vector (second smallest eigenvalue's eigenvector)
    try:
        # Use eigs to get the two smallest eigenvalues
        # For sparse matrices, eigs works well
        vals, vecs = eigs(L, k=2, which='SM')
        # Sort by eigenvalue
        idxs = np.argsort(vals)
        fiedler = vecs[:, idxs[1]].real
    except:
        # Fallback: use power iteration or just use a random vector
        fiedler = np.random.randn(total)
    
    # Normalize Fiedler vector
    fiedler = fiedler / np.linalg.norm(fiedler)
    
    # Target number of clusters: start with 73 (known best)
    target_clusters = 73
    
    # Partition using Fiedler vector: sort by value and split into balanced clusters
    sorted_indices = np.argsort(fiedler)
    cluster_assignments = np.zeros(total, dtype=np.int32)
    cluster_sizes = [total // target_clusters] * target_clusters
    # Distribute remainder
    for i in range(total % target_clusters):
        cluster_sizes[i] += 1
    
    pos = 0
    for c in range(target_clusters):
        cluster_assignments[sorted_indices[pos:pos+cluster_sizes[c]]] = c
        pos += cluster_sizes[c]
    
    # For each cluster, find the best center (minimizes max distance within cluster)
    # A valid covering code center must have the property that all words in its ball
    # are covered. We want each cluster to be contained in the ball of its center.
    
    def find_best_center(cluster_words, current_centers):
        """Find the word that covers the most words in cluster_words."""
        best_center = None
        best_count = -1
        for w in cluster_words:
            ball = balls[w]
            count = sum(1 for x in cluster_words if x in ball)
            if count > best_count:
                best_count = count
                best_center = w
        return best_center, best_count
    
    # Initialize centers
    centers = []
    for c in range(target_clusters):
        cluster_words = [i for i in range(total) if cluster_assignments[i] == c]
        if not cluster_words:
            continue
        center, _ = find_best_center(cluster_words, centers)
        centers.append(center)
    
    # Convert to set
    center_set = set(centers)
    
    # Check coverage
    def check_coverage(center_set):
        covered = set()
        for cw in center_set:
            covered.update(balls[cw])
        return len(covered) == total
    
    # Iterative refinement: Kernighan-Lin style swaps
    # Try to move vertices between clusters to improve coverage
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # First, try to remove redundant centers
        for cw in list(center_set):
            if time.time() > deadline:
                break
            # Check if removing cw leaves all its ball covered
            can_remove = True
            for w in balls[cw]:
                if not any(w in balls[other] for other in center_set if other != cw):
                    can_remove = False
                    break
            if can_remove:
                center_set.remove(cw)
                improved = True
                break  # Restart after removal
        
        # If coverage is broken, repair it
        if not check_coverage(center_set):
            # Find uncovered words
            covered = set()
            for cw in center_set:
                covered.update(balls[cw])
            uncovered = set(range(total)) - covered
            
            # Greedily add centers to cover uncovered words
            for _ in range(len(uncovered)):
                if time.time() > deadline:
                    break
                if not uncovered:
                    break
                # Find the word that covers the most uncovered words
                best_w = None
                best_count = -1
                for w in range(total):
                    if w in center_set:
                        continue
                    count = sum(1 for u in uncovered if u in balls[w])
                    if count > best_count:
                        best_count = count
                        best_w = w
                if best_w is not None:
                    center_set.add(best_w)
                else:
                    break
            
            # After repair, try to remove again
            improved = True
        
        # Try KL-style swaps: move a vertex from one cluster to another
        # This is approximated by trying to replace a center with a better one
        if len(center_set) > 0:
            for cw in list(center_set):
                if time.time() > deadline:
                    break
                # Try replacing cw with each word in its ball
                for new_cw in balls[cw]:
                    if new_cw in center_set:
                        continue
                    new_set = set(center_set)
                    new_set.discard(cw)
                    new_set.add(new_cw)
                    
                    # Check if this improves or maintains coverage
                    if check_coverage(new_set):
                        # Count how many words are now uniquely covered by new_cw
                        # vs cw - we want to minimize redundancy
                        old_unique = 0
                        for w in balls[cw]:
                            if not any(w in balls[other] for other in center_set if other != cw):
                                old_unique += 1
                        
                        new_unique = 0
                        for w in balls[new_cw]:
                            if not any(w in balls[other] for other in new_set if other != new_cw):
                                new_unique += 1
                        
                        # Accept if we reduce unique coverage (more redundancy is better for removal)
                        if new_unique < old_unique:
                            center_set = new_set
                            improved = True
                            break
    
    # Final cleanup pass
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for cw in list(center_set):
                if time.time() > deadline:
                    break
                can_remove = True
                for w in balls[cw]:
                    if not any(w in balls[other] for other in center_set if other != cw):
                        can_remove = False
                        break
                if can_remove:
                    center_set.remove(cw)
                    improved = True
    
    # Verify final coverage
    if not check_coverage(center_set):
        # Fallback: use greedy set cover from scratch
        covered = set()
        center_set = set()
        remaining = set(range(total))
        while remaining and time.time() < deadline:
            best_w = None
            best_count = -1
            for w in remaining:
                count = len(balls[w] & remaining)
                if count > best_count:
                    best_count = count
                    best_w = w
            if best_w is None:
                break
            center_set.add(best_w)
            covered.update(balls[best_w])
            remaining -= covered
    
    # Convert to list of words
    result = [idx_to_word(cw) for cw in center_set]
    
    return result
