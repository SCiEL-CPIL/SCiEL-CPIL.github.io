import time
import random
from itertools import combinations

# Precompute all possible triples on 0..15
ALL_TRIPLES = list(combinations(range(16), 3))
# Precompute bitmasks for all triples
TRIPLE_MASKS = []
for t in ALL_TRIPLES:
    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    TRIPLE_MASKS.append(mask)

# Map triple to index for quick lookup?
# We work with lists of triples, so we can just use the mask.

def calculate_alpha(edges):
    """Find the largest independent set size using bitmasks."""
    # edges is a list of triples (tuples)
    # Convert to masks
    edge_masks = []
    for e in edges:
        mask = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
        edge_masks.append(mask)
        
    # Check sizes from 8 down to 5 (we know 4 is impossible for alpha, but we want the max)
    # Actually, we want the LARGEST independent set.
    # So we check k=8, 7, 6, 5...
    # If we find an independent set of size k, return k.
    
    # Optimization: If we have many edges, alpha is small.
    # We can start from a reasonable upper bound.
    # For 16 points, max alpha is 16 (if no edges).
    # With ~30 edges, alpha is likely 5 or 6.
    
    # To speed up, we can check k=8, 7, 6 first.
    # If none found, check 5.
    
    for k in range(8, 4, -1): # Check 8, 7, 6, 5
        for subset in combinations(range(16), k):
            # Build mask for subset
            smask = 0
            for v in subset:
                smask |= (1 << v)
            
            # Check if any edge is fully contained in subset
            is_independent = True
            for emask in edge_masks:
                if (emask & smask) == emask:
                    is_independent = False
                    break
            
            if is_independent:
                return k
                
    return 4 # Should not happen if logic is correct, but safe fallback

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = []
    best_alpha = 16
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initial Construction: Random Greedy
        shuffled_triples = ALL_TRIPLES[:]
        random.shuffle(shuffled_triples)
        edges = []
        edges_set = set()
        used_pairs = set()
        
        for t in shuffled_triples:
            if len(edges) >= 37: break
            valid = True
            # Check pairs
            for i in range(3):
                for j in range(i+1, 3):
                    p1 = (t[i], t[j])
                    p2 = (t[j], t[i])
                    if p1 in used_pairs or p2 in used_pairs:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                edges.append(t)
                edges_set.add(t)
                for i in range(3):
                    for j in range(i+1, 3):
                        used_pairs.add((t[i], t[j]))
                        used_pairs.add((t[j], t[i]))
        
        current_alpha = calculate_alpha(edges)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_edges = edges[:]
            
        if best_alpha <= 5:
            # Found optimal (likely), can break early if confident
            # But let's try to see if we can get 4 (impossible) or just confirm 5.
            # Since 4 is impossible, 5 is the target.
            # We can break if we find 5.
            break
        
        # Local Search
        # We will perform a few local search steps if we have time
        # But if we already have a good best_alpha, we might skip weak local searches
        if time.time() > deadline - 10:
            continue
            
        # Simple local search: try to swap edges to reduce alpha
        # We only do this if current_alpha > 5
        for step in range(50):
            if time.time() > deadline - 5:
                break
                
            if len(edges) == 0: break
            idx = random.randint(0, len(edges) - 1)
            edge_to_remove = edges[idx]
            
            # Remove edge
            edges.pop(idx)
            edges_set.remove(edge_to_remove)
            for i in range(3):
                for j in range(i+1, 3):
                    used_pairs.discard((edge_to_remove[i], edge_to_remove[j]))
                    used_pairs.discard((edge_to_remove[j], edge_to_remove[i]))
            
            # Find candidate
            candidates = []
            for t in ALL_TRIPLES:
                if t in edges_set: continue
                valid = True
                for i in range(3):
                    for j in range(i+1, 3):
                        p1 = (t[i], t[j])
                        p2 = (t[j], t[i])
                        if p1 in used_pairs or p2 in used_pairs:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    candidates.append(t)
            
            if not candidates:
                # Restore
                edges.append(edge_to_remove)
                edges_set.add(edge_to_remove)
                for i in range(3):
                    for j in range(i+1, 3):
                        used_pairs.add((edge_to_remove[i], edge_to_remove[j]))
                        used_pairs.add((edge_to_remove[j], edge_to_remove[i]))
                continue
                
            new_edge = random.choice(candidates)
            
            # Add new edge
            edges.append(new_edge)
            edges_set.add(new_edge)
            for i in range(3):
                for j in range(i+1, 3):
                    used_pairs.add((new_edge[i], new_edge[j]))
                    used_pairs.add((new_edge[j], new_edge[i]))
            
            new_alpha = calculate_alpha(edges)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_edges = edges[:]
                    if best_alpha <= 5:
                        break
            elif new_alpha == current_alpha:
                if random.random() < 0.5:
                    current_alpha = new_alpha
                else:
                    # Reject
                    edges.pop()
                    edges_set.remove(new_edge)
                    for i in range(3):
                        for j in range(i+1, 3):
                            used_pairs.discard((new_edge[i], new_edge[j]))
                            used_pairs.discard((new_edge[j], new_edge[i]))
                    edges.append(edge_to_remove)
                    edges_set.add(edge_to_remove)
                    for i in range(3):
                        for j in range(i+1, 3):
                            used_pairs.add((edge_to_remove[i], edge_to_remove[j]))
                            used_pairs.add((edge_to_remove[j], edge_to_remove[i]))
            else:
                # Reject
                edges.pop()
                edges_set.remove(new_edge)
                for i in range(3):
                    for j in range(i+1, 3):
                        used_pairs.discard((new_edge[i], new_edge[j]))
                        used_pairs.discard((new_edge[j], new_edge[i]))
                edges.append(edge_to_remove)
                edges_set.add(edge_to_remove)
                for i in range(3):
                    for j in range(i+1, 3):
                        used_pairs.add((edge_to_remove[i], edge_to_remove[j]))
                        used_pairs.add((edge_to_remove[j], edge_to_remove[i]))
                        
        if best_alpha <= 5:
            break

    return [list(e) for e in best_edges]
