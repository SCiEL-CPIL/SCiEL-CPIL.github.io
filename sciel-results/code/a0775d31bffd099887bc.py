<<<<<<< SEARCH
def _better_greedy(n, adj, cliques, independents):
    """Improved greedy that tries multiple vertex orderings."""
    best_parts = None
    best_count = n + 1
    
    # Try several orderings
    orderings = [
        list(range(n)),
        sorted(range(n), key=lambda v: len(adj[v])),  # by degree
        sorted(range(n), key=lambda v: -len(adj[v])),  # by degree desc
    ]
    
    for ordering in orderings:
        parts = []
        part_masks = []
        for v in ordering:
            best_part = None
            best_gain = -1
            for i, pmask in enumerate(part_masks):
                trial_mask = pmask | (1 << v)
                if cliques[trial_mask] or independents[trial_mask]:
                    gain = bin(trial_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_part = i
            if best_part is not None:
                part_masks[best_part] |= (1 << v)
            else:
                part_masks.append(1 << v)
        
        count = len(part_masks)
        if count < best_count:
            best_count = count
            best_parts = [ [i for i in range(n) if pmask & (1 << i)] for pmask in part_masks if pmask != 0 ]
    
    return best_parts
=======
def _better_greedy(n, adj, cliques, independents):
    """Improved greedy that tries multiple vertex orderings with local search."""
    best_parts = None
    best_count = n + 1
    
    # Try several orderings
    orderings = [
        list(range(n)),
        sorted(range(n), key=lambda v: len(adj[v])),  # by degree
        sorted(range(n), key=lambda v: -len(adj[v])),  # by degree desc
        sorted(range(n), key=lambda v: -bin(len(adj[v])).count('1')),  # by degree bit count
    ]
    
    for ordering in orderings:
        part_masks = []
        for v in ordering:
            best_part = None
            best_gain = -1
            for i, pmask in enumerate(part_masks):
                trial_mask = pmask | (1 << v)
                if cliques[trial_mask] or independents[trial_mask]:
                    gain = bin(trial_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_part = i
            if best_part is not None:
                part_masks[best_part] |= (1 << v)
            else:
                part_masks.append(1 << v)
        
        count = len(part_masks)
        if count < best_count:
            best_count = count
            best_parts = [ [i for i in range(n) if pmask & (1 << i)] for pmask in part_masks if pmask != 0 ]
    
    # Local search: try to merge or move vertices
    if best_parts is not None and best_count > 1:
        improved = True
        while improved:
            improved = False
            new_count = best_count
            
            # Try moving each vertex to another part if it fits
            for part_idx, part in enumerate(best_parts):
                for v in list(part):
                    for other_idx, other_part in enumerate(best_parts):
                        if other_idx == part_idx:
                            continue
                        # Check if v can be added to other_part
                        trial_mask = 0
                        for u in other_part:
                            trial_mask |= (1 << u)
                        trial_mask |= (1 << v)
                        if cliques[trial_mask] or independents[trial_mask]:
                            # Move v to other part
                            part.remove(v)
                            other_part.append(v)
                            improved = True
                            break
                    if improved:
                        break
            
            # Rebuild best_parts from non-empty parts
            new_parts = [p for p in best_parts if p]
            if len(new_parts) < best_count:
                best_parts = new_parts
                best_count = len(new_parts)
                improved = True
    
    return best_parts
>>>>>>> REPLACE

