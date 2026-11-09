"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Convert to int8 for faster matrix multiplication and memory efficiency
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin (10s buffer)
    deadline = time.time() + 140.0
    
    # Helper: Calculate cut size efficiently
    # Cut size = (sum of edges) - (sum of edges within same partition)
    # Alternatively: 0.5 * sum(|labels_i - labels_j| * adj_ij)
    # Even faster for binary labels: 
    # Let L be the label vector (0 or 1).
    # Edges crossing = sum over i,j where L_i != L_j of A_ij
    # Note that sum_{i,j} A_ij (1 - L_i L_j - L_i(1-L_j) - (1-L_i)L_j) is complex.
    # Simpler: 
    # Let S = L (vector of 0/1).
    # Edges in cut = sum_{i,j: L_i != L_j} A_ij
    # We can compute: 
    # total_edges = 0.5 * sum(A)
    # intra_edges = 0.5 * (S^T A S + (1-S)^T A (1-S))
    # cut = total_edges - intra_edges
    
    def _cut_size(labels):
        # labels is 1D array of 0/1
        # S = labels
        # intra = 0.5 * (S @ A @ S + (1-S) @ A @ (1-S))
        # This is O(N^2) per call. For batch SA, we need to update incrementally or use batch ops.
        # For initialization and final check, this is fine.
        S = labels.astype(np.int8)
        # A is symmetric
        intra1 = S @ (adj @ S)
        S_inv = 1 - S
        intra2 = S_inv @ (adj @ S_inv)
        intra = 0.5 * (intra1 + intra2)
        total_edges = 0.5 * adj.sum()
        return total_edges - intra

    # Helper: Spectral Initialization (Goemans-Williamson style)
    def _spectral_init():
        # Compute Laplacian L = D - A
        # D is degree matrix
        degrees = adj.sum(axis=1)
        # Avoid building full D matrix if possible, but N=100 is small.
        D = np.diag(degrees)
        L = D - adj.astype(np.float64)
        
        # Find Fiedler vector (2nd smallest eigenvalue)
        # For N=100, full eigendecomposition is fast enough.
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            
            # Threshold at mean to get 0/1 labels
            mean_val = np.mean(fiedler)
            labels = (fiedler >= mean_val).astype(np.int8)
            
            # Ensure balanced-ish partition? Not strictly required but helps.
            # If all 0s or all 1s, fallback to random
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, n).astype(np.int8)
            return labels
        except:
            return np.random.randint(0, 2, n).astype(np.int8)

    # Helper: Vectorized Batch Simulated Annealing Step
    # We want to evaluate flipping a subset of nodes.
    # Delta in cut size when flipping node i:
    # If we flip i, edges (i,j) change state.
    # Current contribution of i to cut: sum_{j: L_j != L_i} A_ij
    # New contribution: sum_{j: L_j == L_i} A_ij (since L_i flips, condition inverts)
    # Delta_i = New - Old = (sum_{j: L_j == L_i} A_ij) - (sum_{j: L_j != L_i} A_ij)
    # Let same = sum_{j: L_j == L_i} A_ij
    # Let diff = sum_{j: L_j != L_i} A_ij
    # Delta_i = same - diff
    # Note: same + diff = degree_i
    # Delta_i = same - (degree_i - same) = 2*same - degree_i
    
    def _batch_sa_step(labels, temp, n_batch=10):
        # Select a batch of nodes to potentially flip
        # We calculate the delta for ALL nodes, then randomly pick some to flip based on SA criteria
        # This is more efficient than sequential.
        
        # 1. Compute current 'same' partition sums for all nodes
        # same_i = sum_{j: L_j == L_i} A_ij
        # We can compute this using matrix multiplication with masks, or:
        # Let S = labels (0/1)
        # For node i with label 0: same_i = sum_{j: L_j == 0} A_ij
        # For node i with label 1: same_i = sum_{j: L_j == 1} A_ij
        
        # Efficient way:
        # Create mask matrices? No, N=100 is small.
        # Let's compute the "agreement" matrix.
        # Agreement[i,j] = 1 if L_i == L_j else 0.
        # Same_counts[i] = sum_j A[i,j] * Agreement[i,j]
        
        # Vectorized agreement calculation:
        # L is (N, 1). 
        # L_eq = (L == L.T) -> (N, N) boolean matrix
        # This is O(N^2) memory and time. N=100 -> 10k ops. Very fast.
        L_col = labels.reshape(-1, 1)
        agreement = (L_col == L_col.T)
        
        # Same_counts = (adj * agreement).sum(axis=1)
        same_counts = (adj * agreement).sum(axis=1)
        
        degrees = adj.sum(axis=1)
        
        # Delta if we flip node i:
        # Delta_i = 2 * same_counts[i] - degrees[i]
        deltas = 2 * same_counts - degrees
        
        # 2. Select a random subset of nodes to consider flipping
        # We consider all nodes, but only accept with probability based on delta.
        # To keep it "batch", we can evaluate all, decide flips for all, then apply.
        # However, flipping multiple nodes simultaneously changes the context for others.
        # Standard SA flips one at a time. 
        # "Batch SA" usually means: propose a set of flips, calculate total delta, accept/reject.
        # Or: calculate individual deltas, flip those with delta > 0 immediately (greedy), and use SA for the rest?
        # Let's stick to a robust approach: 
        # Iterate through nodes in random order, but use vectorized calculation of deltas for the *current* state.
        # Since N=100, even sequential is fast. The bottleneck is Python overhead.
        # Let's try a "Sweep" approach: 
        # 1. Calculate deltas for all nodes in current state.
        # 2. Randomly shuffle nodes.
        # 3. For each node in shuffle:
        #    - Re-calculate delta for this node? No, if we flip previous nodes, delta changes.
        #    - If we don't re-calculate, we are doing a "batch" update which is an approximation.
        
        # Better: Since N is small, let's do a fast sequential sweep but vectorized where possible?
        # Actually, the most efficient Python approach for N=100 is to just do the sequential loop but minimize Python overhead.
        # The current seed does 3 passes. We can do many more passes in the same time.
        
        # Let's implement a fast sequential local search with SA acceptance.
        # To make it faster than pure Python loop, we can use NumPy for the "same/diff" calculation per node?
        # No, that's still a loop.
        
        # Alternative: Use the vectorized delta calculation to find ALL improving moves, flip them, repeat.
        # This is "Parallel Greedy" or "Simultaneous Update".
        # It converges faster but might oscillate. SA helps escape.
        
        # Let's try: 
        # 1. Calculate deltas for all.
        # 2. Identify nodes where delta > 0 (improving moves).
        # 3. Flip all of them simultaneously.
        # 4. If no improving moves, use SA to flip a random node with delta < 0 (probabilistic acceptance).
        
        # This "Simultaneous Greedy" is very fast per iteration.
        
        # Step 2a: Find improving nodes
        improving_mask = deltas > 0
        if np.any(improving_mask):
            labels[improving_mask] = 1 - labels[improving_mask]
            return True # State changed
        
        # Step 2b: No improving moves. Try SA move.
        # Pick a random node
        v = np.random.randint(n)
        delta_v = deltas[v]
        
        # Accept if delta > 0 (already handled) or with prob exp(delta/T)
        if delta_v < 0:
            if np.random.rand() < np.exp(delta_v / temp):
                labels[v] = 1 - labels[v]
                return True
        
        return False # No change

    # Main Loop
    best_labels = None
    best_score = -1
    
    # Run multiple restarts to explore different basins
    # Time budget allows for many restarts of a fast local search.
    
    # First, try Spectral Init + Fast Local Search
    init_labels = _spectral_init()
    
    # Local Search Phase
    # We will run the simultaneous greedy/SA step until no change or time limit.
    temp = 1.0
    min_temp = 0.01
    
    current_labels = init_labels.copy()
    current_score = _cut_size(current_labels)
    if current_score > best_score:
        best_score = current_score
        best_labels = current_labels.copy()
    
    # Cooling schedule
    # We don't know how many steps will take. 
    # Let's define a fixed number of "rounds" or use time.
    # Since N=100, one "round" (calculate deltas + update) is very fast (~1ms).
    # We can do thousands of rounds.
    
    steps = 0
    while time.time() < deadline - 5.0: # Leave 5s for other restarts
        changed = _batch_sa_step(current_labels, temp, n_batch=n)
        steps += 1
        
        # Cool down
        if steps % 10 == 0:
            temp *= 0.95
            if temp < min_temp:
                temp = min_temp
        
        # Check score periodically? 
        # The step function only accepts improving moves or SA moves.
        # If SA move is accepted, score might drop. 
        # We should track the best score seen so far.
        # But calculating score every step is O(N^2). 
        # We can update score incrementally?
        # In _batch_sa_step, we know the deltas.
        # If we flip a set of nodes, the score change is not simply sum of deltas due to interactions.
        # So we must recalculate score if we want to track "best".
        # However, since we mostly do greedy flips (delta > 0), the score usually increases.
        # For SA moves, it might decrease.
        # Let's recalculate score every 10 steps to save time.
        
        if steps % 10 == 0:
            s = _cut_size(current_labels)
            if s > best_score:
                best_score = s
                best_labels = current_labels.copy()
        
        # If we hit a local optimum (no improving moves) and SA is cold, we might stagnate.
        # The _batch_sa_step returns False if no change.
        if not changed:
            # Try a random restart or a larger SA move?
            # Let's just break this local search phase and start a new one with different init.
            break

    # Check final score of this phase
    final_s = _cut_size(current_labels)
    if final_s > best_score:
        best_score = final_s
        best_labels = current_labels.copy()

    # If time remains, do a few more random restarts with spectral init
    # Or just random init? Spectral is usually better.
    while time.time() < deadline - 2.0:
        # New restart
        # Mix: 50% Spectral, 50% Random
        if np.random.rand() < 0.5:
            current_labels = _spectral_init()
        else:
            current_labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Quick local search
        temp = 1.0
        steps = 0
        while time.time() < deadline - 2.0:
            changed = _batch_sa_step(current_labels, temp)
            steps += 1
            if steps % 10 == 0:
                temp *= 0.95
                if temp < min_temp:
                    temp = min_temp
                s = _cut_size(current_labels)
                if s > best_score:
                    best_score = s
                    best_labels = current_labels.copy()
            if not changed:
                break
        
        final_s = _cut_size(current_labels)
        if final_s > best_score:
            best_score = final_s
            best_labels = current_labels.copy()

    return best_labels

# EVOLVE-BLOCK-END
