"""Maximum Cut: spectral/SDP init + fast local search + perturbation search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    # Convert to numpy int array
    adj = np.array(adjacency, dtype=np.int32)
    # Precompute degree and neighbor lists for fast delta computation
    degrees = adj.sum(axis=1)
    # For fast flip delta: gain(v) = 2*sum(adj[v, other_side]) - degrees[v]
    # Actually for flip: new_cut - old_cut = 2*sum_{u in other_side} adj[v,u] - sum_{u in same_side} adj[v,u]
    # = 2*sum_{u in other_side} adj[v,u] - (degrees[v] - sum_{u in same_side} adj[v,u])
    # Let's just use: delta_flip(v) = 2 * (sum adj[v, labels!=labels[v]]) - degrees[v]
    # No wait: if we flip v from label 0 to 1:
    #   Edges from v to label-1 nodes: were cut, now not cut -> lose those
    #   Edges from v to label-0 nodes: were not cut, now cut -> gain those
    #   So delta = (num neighbors with label 0) - (num neighbors with label 1)
    #   = same_side_count - other_side_count where same_side = same label as v before flip
    #   Actually: delta = (neighbors with opposite label) - (neighbors with same label)
    #   = other_side - same_side

    def compute_cut(labels):
        """Compute cut size using vectorized operation."""
        diff = (labels[:, None] != labels[None, :]).astype(np.int32)
        return int(np.sum(adj * diff) // 2)

    def flip_greedy(labels, max_passes=50):
        """Single-flip greedy improvement with fast delta computation."""
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            # Randomize order for diversity
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    break
                # delta = (neighbors with opposite label) - (neighbors with same label)
                same_side = adj[v, labels == labels[v]].sum()
                other_side = adj[v, labels != labels[v]].sum()
                if other_side > same_side:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def pairwise_swap_search(labels, max_passes=20):
        """2-swap local search with fast delta computation."""
        labels = labels.copy()
        current_cut = compute_cut(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for idx_i in range(n):
                if time.time() > deadline:
                    break
                i = order[idx_i]
                for idx_j in range(idx_i + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[idx_j]
                    if labels[i] == labels[j]:
                        continue

                    # Compute delta for swapping i and j
                    # Swapping means: i gets j's label, j gets i's label
                    # If labels[i]=0, labels[j]=1:
                    #   i: was 0, becomes 1. Delta_i = (neighbors of i with label 1) - (neighbors of i with label 0)
                    #   j: was 1, becomes 0. Delta_j = (neighbors of j with label 0) - (neighbors of j with label 1)
                    # But we need to account for the edge between i and j
                    # Let's compute it properly:
                    # Before: i has label a, j has label b (a != b)
                    # After: i has label b, j has label a
                    # 
                    # For vertex i (changing from a to b):
                    #   Edges to nodes with label a: were not cut, now cut -> gain count_a
                    #   Edges to nodes with label b: were cut, now not cut -> lose count_b
                    #   But the edge (i,j) itself: was cut (a!=b), still cut (b!=a) -> no change
                    #   So delta_i = count_a_excl_j - count_b_excl_j
                    #   where count_a_excl_j = sum adj[i, labels==a] - adj[i,j]
                    #   count_b_excl_j = sum adj[i, labels==b] - adj[i,j]
                    #   delta_i = (sum adj[i, labels==a] - adj[i,j]) - (sum adj[i, labels==b] - adj[i,j])
                    #          = sum adj[i, labels==a] - sum adj[i, labels==b]
                    #
                    # Similarly for vertex j (changing from b to a):
                    #   delta_j = sum adj[j, labels==b] - sum adj[j, labels==a]
                    #
                    # Total delta = delta_i + delta_j
                    # But wait, we're double-counting the edge (i,j)?
                    # Let me think again more carefully.
                    #
                    # Actually, let's just compute:
                    # delta = 2 * (adj[i,j] if labels[i]!=labels[j] else 0) ... no this is getting complex.
                    #
                    # Simpler: just compute the cut change directly for the affected edges.
                    # The only edges whose cut status can change are those incident to i or j.
                    # 
                    # Let's use a simpler approach: compute delta by checking neighbors
                    li, lj = labels[i], labels[j]
                    # For i: neighbors with label lj (opposite to current li) become same, neighbors with label li become opposite
                    # Gain for i: neighbors with label li (same as i) that become opposite -> they become cut
                    # Lose for i: neighbors with label lj (opposite to i) that become same -> they become uncut
                    # But exclude j from both since j is also changing
                    gain_i = adj[i, labels == li].sum() - adj[i, j]  # neighbors same as i, excluding j
                    lose_i = adj[i, labels == lj].sum() - adj[i, j]  # neighbors opposite to i, excluding j
                    # For j: neighbors with label lj (same as j) that become opposite -> they become cut
                    # Lose for j: neighbors with label li (opposite to j) that become same -> they become uncut
                    gain_j = adj[j, labels == lj].sum() - adj[i, j]  # neighbors same as j, excluding i
                    lose_j = adj[j, labels == li].sum() - adj[i, j]  # neighbors opposite to j, excluding i
                    # Edge (i,j): was cut (li != lj), after swap: j has li, i has lj, still li != lj, still cut
                    # So edge (i,j) contributes 0 to delta
                    delta = (gain_i + gain_j) - (lose_i + lose_j)

                    if delta > 0:
                        labels[i], labels[j] = labels[j], labels[i]
                        current_cut += delta
                        improved = True

            if not improved:
                break
        return labels, current_cut

    def spectral_init():
        """Spectral initialization using Fiedler vector."""
        try:
            degrees = adj.sum(axis=1)
            L = np.diag(degrees) - adj
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            labels = (fiedler > 0).astype(np.int32)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def sdp_hyperplane_init():
        """SDP-inspired hyperplane initialization."""
        try:
            k = min(5, n)
            eigvals, eigvecs = np.linalg.eigh(adj)
            idx = np.argsort(eigvals)[::-1]
            top_vecs = eigvecs[:, idx[:k]]
            random_coeff = np.random.randn(k)
            scores = top_vecs @ random_coeff
            labels = (scores > 0).astype(np.int32)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, size=n)
        except Exception:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def random_init():
        return np.random.randint(0, 2, size=n)

    def perturbation_search(labels, num_perturbs=50):
        """Perturb the best solution and re-optimize to escape local optima."""
        best_labels = labels.copy()
        best_cut = compute_cut(best_labels)

        for p in range(num_perturbs):
            if time.time() > deadline - 10:
                break

            # Create a perturbed copy: flip a random subset of vertices
            cand = best_labels.copy()
            # Flip between 5% and 30% of vertices
            frac = random.uniform(0.05, 0.30)
            num_flips = max(1, int(n * frac))
            flip_idx = np.random.choice(n, size=num_flips, replace=False)
            cand[flip_idx] = 1 - cand[flip_idx]

            # Local search on perturbed solution
            cand = flip_greedy(cand, max_passes=30)
            cand, cut = pairwise_swap_search(cand, max_passes=15)

            if cut > best_cut:
                best_cut = cut
                best_labels = cand.copy()

        return best_labels, best_cut

    best_labels = None
    best_cut = -1

    # 1. SDP hyperplane init
    if time.time() < deadline:
        labels = sdp_hyperplane_init()
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 2. Spectral init
    if time.time() < deadline:
        labels = spectral_init()
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 3. Alternating parity init
    if time.time() < deadline:
        labels = np.arange(n) % 2
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 4. Random restarts
    num_random_restarts = 8
    for r in range(num_random_restarts):
        if time.time() > deadline - 30:
            break
        labels = random_init()
        labels = flip_greedy(labels, max_passes=20)
        labels, cut = pairwise_swap_search(labels, max_passes=15)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 5. Perturbation search on best solution
    if best_labels is not None and time.time() < deadline - 15:
        best_labels, best_cut = perturbation_search(best_labels, num_perturbs=100)

    # 6. Final polish: more swap passes on the best solution
    if best_labels is not None and time.time() < deadline - 5:
        labels, cut = pairwise_swap_search(best_labels, max_passes=50)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
