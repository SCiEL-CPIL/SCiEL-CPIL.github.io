import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Generate weight-4 vectors
    vecs4 = []
    for support in itertools.combinations(range(D), 4):
        for mask in range(16):
            pos = 0
            neg = 0
            for i, idx in enumerate(support):
                if mask & (1 << i):
                    pos |= (1 << idx)
                else:
                    neg |= (1 << idx)
            vecs4.append((pos, neg))
            
    N4 = len(vecs4)
    
    # Helper: dot product
    def dot4(pos1, neg1, pos2, neg2):
        return (pos1 & pos2).bit_count() - (pos1 & neg2).bit_count() - (neg1 & pos2).bit_count() + (neg1 & neg2).bit_count()
        
    # Build compatibility graph for weight-4 vectors
    # Edge if dot <= 2
    # Use bitsets for efficiency
    # 5280 nodes -> 83 uint64s per node
    num_words = (N4 + 63) // 64
    adj = [0] * (N4 * num_words)
    
    for i in range(N4):
        p1, n1 = vecs4[i]
        for j in range(i + 1, N4):
            p2, n2 = vecs4[j]
            if dot4(p1, n1, p2, n2) <= 2:
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i * num_words + (j >> 6)] |= (1 << (j & 63))
                adj[j * num_words + (i >> 6)] |= (1 << (i & 63))
                
    # Maximum Clique Algorithm (Tomita's MCQ simplified)
    # We use a recursive search with coloring bound
    
    best_clique = []
    
    # Order nodes by degree (descending) to improve pruning
    degrees = [0] * N4
    for i in range(N4):
        deg = 0
        base = i * num_words
        for w in range(num_words):
            deg += bin(adj[base + w]).count('1')
        degrees[i] = deg
        
    sorted_indices = sorted(range(N4), key=lambda x: degrees[x], reverse=True)
    
    # Map original index to sorted position
    # We will work on the sorted graph
    # Reconstruct adjacency in sorted order
    # This is expensive, so we might stick to original order but use sorted order for branching?
    # Or just use the original order and pick the best candidate based on degree in the current candidate set.
    
    # Let's use a simpler greedy + local search approach first, as Tomita might be complex to implement correctly under time pressure.
    # Greedy MIS with random restarts.
    
    def greedy_mis(indices):
        selected = []
        remaining = set(indices)
        while remaining:
            # Pick node with minimum degree in the remaining induced subgraph?
            # Or just pick the first one?
            # Let's pick the one with minimum degree in the *conflict* graph (maximum degree in compatibility).
            # Actually, for MIS, we want to pick nodes that conflict with few others.
            # Conflict graph edge if dot > 2.
            # So we want min degree in conflict graph.
            
            # Calculate degrees in remaining set
            # This is O(N^2) per step, too slow.
            
            # Let's stick to the pre-computed degree heuristic.
            # Pick the node with lowest global degree from remaining.
            min_deg_node = None
            min_deg = float('inf')
            for node in remaining:
                if degrees[node] < min_deg:
                    min_deg = degrees[node]
                    min_deg_node = node
            
            if min_deg_node is None:
                break
                
            selected.append(min_deg_node)
            
            # Remove min_deg_node and its neighbors in the *conflict* graph
            # Neighbors in conflict graph are those with dot > 2
            # In compatibility graph, they are NOT neighbors.
            # So we remove min_deg_node and all nodes NOT connected to it in compatibility graph?
            # No, in MIS, if we select a node, we remove it and its *neighbors in the original graph*.
            # Here, the "original graph" for MIS is the Conflict Graph.
            # Conflict Graph edge (i,j) if dot(i,j) > 2.
            # So if we select i, we remove all j such that dot(i,j) > 2.
            
            remove_set = set()
            p1, n1 = vecs4[min_deg_node]
            for j in remaining:
                if j == min_deg_node:
                    continue
                p2, n2 = vecs4[j]
                if dot4(p1, n1, p2, n2) > 2:
                    remove_set.add(j)
            
            remaining -= remove_set
            remaining.discard(min_deg_node)
            
        return selected

    best_size = 0
    best_selected = []
    
    # Try a few random permutations of the initial order
    # Or just run greedy a few times with slight variations
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Shuffle the order of consideration?
        # The greedy picks min degree. If we shuffle, the "min degree" calculation is static.
        # To make it dynamic, we need to recalculate degrees or use a different heuristic.
        
        # Let's try a different heuristic: Random Greedy
        # Pick a random node, add to MIS, remove conflicts.
        
        selected = []
        remaining = set(range(N4))
        
        while remaining and time.time() < deadline:
            # Pick a random node
            node = random.choice(list(remaining))
            selected.append(node)
            
            # Remove conflicts
            p1, n1 = vecs4[node]
            remove_set = set()
            for j in remaining:
                if j == node:
                    continue
                p2, n2 = vecs4[j]
                if dot4(p1, n1, p2, n2) > 2:
                    remove_set.add(j)
            
            remaining -= remove_set
            remaining.discard(node)
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()

    # Convert best_selected to vectors
    selected4 = [vecs4[i] for i in best_selected]
    
    # Add weight-2 vectors
    vecs2 = []
    for support in itertools.combinations(range(D), 2):
        for mask in range(4):
            pos = 0
            neg = 0
            for i, idx in enumerate(support):
                if mask & (1 << i):
                    pos |= (1 << idx)
                else:
                    neg |= (1 << idx)
            vecs2.append((pos, neg))
            
    def dot2(pos1, neg1, pos2, neg2):
        return (pos1 & pos2).bit_count() - (pos1 & neg2).bit_count() - (neg1 & pos2).bit_count() + (neg1 & neg2).bit_count()

    selected2 = []
    for p, n in vecs2:
        if time.time() > deadline:
            break
        ok4 = True
        for p_s, n_s in selected4:
            if dot2(p, n, p_s, n_s) > 1:
                ok4 = False
                break
        if not ok4:
            continue
            
        ok2 = True
        for p_s, n_s in selected2:
            if dot2(p, n, p_s, n_s) > 0:
                ok2 = False
                break
        if ok2:
            selected2.append((p, n))
            
    # Construct output
    total_k = len(selected4) + len(selected2)
    result = np.zeros((total_k, D), dtype=np.int64)
    
    k = 0
    for p, n in selected4:
        for i in range(D):
            if p & (1 << i):
                result[k, i] = 1
            elif n & (1 << i):
                result[k, i] = -1
        k += 1
        
    for p, n in selected2:
        for i in range(D):
            if p & (1 << i):
                result[k, i] = 1
            elif n & (1 << i):
                result[k, i] = -1
        k += 1
        
    return result
