<<<<<<< SEARCH
def _local_search(S, max_iters=5000, deadline=None):
    """Try to improve S by swapping or adding elements."""
    S = set(S)
    best_size = len(S)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try adding any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflict_pairs_with_x(x, S):
                S.add(x)
                best_size = max(best_size, len(S))
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for a in list(S):
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflict_pairs_with_x(x, S):
                    S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        improved = True
                    break
            if not improved:
                S.add(a)
        
        if not improved:
            break
    
    return S
=======
def _local_search(S, max_iters=5000, deadline=None):
    """Try to improve S by swapping or adding elements."""
    S = set(S)
    best_size = len(S)
    stuck_count = 0
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try adding any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflict_pairs_with_x_fast(x, S):
                S.add(x)
                best_size = max(best_size, len(S))
                improved = True
                stuck_count = 0
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove one, add another
        found_swap = False
        for a in list(S):
            S.remove(a)
            # Try to add the best candidate
            best_x = None
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflict_pairs_with_x_fast(x, S):
                    best_x = x
                    break
            if best_x is not None:
                S.add(best_x)
                if len(S) > best_size:
                    best_size = len(S)
                    improved = True
                    stuck_count = 0
                    found_swap = True
                else:
                    # Keep the swap even if size didn't increase, to escape local optima
                    S.add(best_x)
                    found_swap = True
                    stuck_count = 0
            else:
                S.add(a)
            
            if improved:
                break
        
        if not improved:
            stuck_count += 1
            if stuck_count > 10:
                break
        else:
            stuck_count = 0
    
    return S
>>>>>>> REPLACE

