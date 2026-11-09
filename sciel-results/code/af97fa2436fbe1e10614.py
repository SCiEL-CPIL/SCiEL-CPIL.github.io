"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _greedy_local_search(adjacency, labels, gains, deadline):
    """
    Perform greedy hill-climbing until no single flip improves the cut.
    Returns updated labels, gains, and cut size.
    """
    n = len(labels)
    current_cut = _cut_size(adjacency, labels)
    
    # We use a queue of vertices to check. 
    # To be efficient, we can just iterate through all vertices repeatedly 
    # until no improvement is found, but that's O(N^2) per pass.
    # Better: maintain a list of "active" vertices.
    # For simplicity and speed in Python, a simple loop over all vertices 
    # checking if gain > 0 is often fast enough for N=100 if we limit passes.
    
    # However, standard local search for MaxCut:
    # Flip v if gain[v] > 0.
    # Update gains for v and neighbors.
    
    improved = True
    # Limit the number of passes to avoid infinite loops in case of bugs, 
    # though greedy on finite discrete space must terminate.
    max_passes = 100
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        
        # We can randomize the order of checking to break symmetries, 
        # but deterministic order is fine for greedy.
        # To speed up, we can check all vertices.
        for v in range(n):
            if gains[v] > 0:
                # Accept flip
                current_cut += gains[v]
                
                # Update labels
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update gains
                # v's gain becomes -gain[v]
                gains[v] = -gains[v]
                
                # Update neighbors
                neighbors = np.where(adjacency[v])[0]
                for u in neighbors:
                    # If u had same label as old v, u's gain increases by 2
                    # If u had diff label as old v, u's gain decreases by 2
                    # current_labels[u] is unchanged.
                    # old_label was labels[v] before flip.
                    if labels[u] == old_label:
                        gains[u] += 2
                    else:
                        gains[u] -= 2
                
                improved = True
                # Early exit if we hit deadline? 
                # Greedy steps are very fast. 
                if time.time() > deadline:
                    return labels, gains, current_cut
    
    return labels, gains, current_cut

def _init_gains(adjacency, labels):
    """Calculate initial gain array."""
    diff_mat = labels[None, :] != labels[:, None]
    other_counts = (adjacency & diff_mat).sum(axis=1)
    same_counts = (adjacency & ~diff_mat).sum(axis=1)
    return other_counts - same_counts

def partition(adjacency):
    """
    Main partition function using Parallel Tempering / Multi-Start Greedy.
    """
    n = len(adjacency)
    # Safety margin: Evaluator kills at 150s. We aim for 140s.
    deadline = time.time() + 140.0
    
    # Number of replicas (parallel searches)
    # N=100 is small. We can run many replicas.
    # Let's try 10 replicas.
    num_replicas = 10
    
    # Initialize replicas with random labels
    # Using different random seeds or just random calls
    replicas = []
    for i in range(num_replicas):
        labels = np.random.randint(0, 2, size=n)
        gains = _init_gains(adjacency, labels)
        replicas.append({
            'labels': labels,
            'gains': gains,
            'cut': _cut_size(adjacency, labels)
        })
    
    best_global_cut = -1
    best_global_labels = None
    
    # Parallel Tempering Loop
    # We alternate between:
    # 1. Local Search (Greedy) on all replicas
    # 2. Exchange (Swap) between replicas to escape local optima
    
    # How often to exchange?
    # If we exchange too often, we lose the benefit of local search.
    # If too rarely, we get stuck.
    # Let's do a local search pass, then an exchange step.
    
    # To make it more effective, we can perform multiple local search steps 
    # before an exchange, or just one.
    # Given N=100, one greedy pass is very fast (<1ms).
    # So we can afford to do many iterations.
    
    while time.time() < deadline:
        # Phase 1: Local Search on all replicas
        for i in range(num_replicas):
            if time.time() > deadline:
                break
            labels, gains, cut = _greedy_local_search(
                adjacency, replicas[i]['labels'], replicas[i]['gains'], deadline
            )
            replicas[i]['labels'] = labels
            replicas[i]['gains'] = gains
            replicas[i]['cut'] = cut
            
            if cut > best_global_cut:
                best_global_cut = cut
                best_global_labels = labels.copy()
        
        if time.time() > deadline:
            break
            
        # Phase 2: Exchange (Tempering)
        # Strategy: Sort replicas by cut size.
        # Swap the best and the worst? Or swap random pairs?
        # Standard Parallel Tempering swaps based on temperature.
        # Here, we can just swap the state of the best and worst replica.
        # This allows the "best" configuration to be perturbed by the "worst" 
        # (which is likely in a different basin), and the worst to inherit the good structure.
        
        # Actually, a simpler and effective heuristic for MaxCut:
        # Identify the replica with the highest cut and the one with the lowest.
        # Swap their labels and gains.
        
        # To add some stochasticity, we can also just do a random swap 
        # between two distinct replicas.
        
        # Let's try: Sort by cut. Swap index 0 (best) and index num_replicas-1 (worst).
        # Also, to prevent cycling, we can randomize the order of replicas 
        # slightly or just accept the swap.
        
        indices = np.argsort([r['cut'] for r in replicas])
        best_idx = indices[-1]
        worst_idx = indices[0]
        
        if best_idx != worst_idx:
            # Swap labels and gains
            replicas[best_idx]['labels'], replicas[worst_idx]['labels'] = \
                replicas[worst_idx]['labels'], replicas[best_idx]['labels']
            replicas[best_idx]['gains'], replicas[worst_idx]['gains'] = \
                replicas[worst_idx]['gains'], replicas[best_idx]['gains']
            replicas[best_idx]['cut'], replicas[worst_idx]['cut'] = \
                replicas[worst_idx]['cut'], replicas[best_idx]['cut']
            
            # Note: We don't need to update best_global here because 
            # the values are just swapped, and we track the global max 
            # in the local search phase. However, if the "worst" was actually 
            # the global best from a previous iteration that got demoted? 
            # No, best_global is a separate variable.
            
        # Optional: Random perturbation?
        # If we just swap, we might cycle. 
        # Adding a small random flip to the "worst" replica before swapping 
        # can help.
        # Let's flip a random vertex in the worst replica (now holding the old best's state? No, they swapped).
        # Actually, after swap, the replica at worst_idx holds the old best's labels.
        # Let's flip a random vertex in the replica at worst_idx to perturb it.
        if num_replicas > 1:
            v = np.random.randint(n)
            # Flip v in the worst replica
            r = replicas[worst_idx]
            old_label = r['labels'][v]
            r['labels'][v] = 1 - old_label
            r['gains'][v] = -r['gains'][v]
            neighbors = np.where(adjacency[v])[0]
            for u in neighbors:
                if r['labels'][u] == old_label:
                    r['gains'][u] += 2
                else:
                    r['gains'][u] -= 2
            r['cut'] += r['gains'][v] # Wait, gain[v] was updated to -gain. 
            # Let's recompute cut for safety if perturbation logic is tricky.
            # Or just calculate delta correctly.
            # Delta = -old_gain[v]. 
            # We updated gains[v] to -gains[v]. So new gain is -old_gain.
            # The cut change is -old_gain.
            # Let's just recalculate cut for the perturbed replica to be safe.
            r['cut'] = _cut_size(adjacency, r['labels'])

    return best_global_labels
# EVOLVE-BLOCK-END
