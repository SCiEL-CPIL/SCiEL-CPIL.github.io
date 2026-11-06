<<<<<<< SEARCH
def _greedy_construct(order):
    """
    Construct a valid set by greedily adding elements in the given order.
    """
    S = []
    for x in order:
        candidate = S + [x]
        if _is_valid(candidate):
            S.append(x)
    return S
=======
def _greedy_construct(order):
    """
    Construct a valid set by greedily adding elements in the given order.
    """
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        candidate = S + [x]
        if _is_valid(candidate):
            S.append(x)
            S_set.add(x)
    return S


def _local_search(S, rng, max_swaps=50):
    """
    Try to improve set S by swapping elements or adding/removing.
    Returns an improved set if found.
    """
    S = list(S)
    S_set = set(S)
    all_elements = set(range(1, 25))
    not_in_S = list(all_elements - S_set)
    
    best = S
    best_len = len(S)
    
    # Try adding elements
    for x in not_in_S:
        candidate = S + [x]
        if _is_valid(candidate):
            S = candidate
            S_set.add(x)
            not_in_S.remove(x)
            if len(S) > best_len:
                best = S[:]
                best_len = len(S)
    
    # Try swaps: remove one, add another
    for _ in range(max_swaps):
        if _time_remaining() < 1.0:
            break
        improved = False
        for i in range(len(S)):
            for j in not_in_S:
                candidate = S[:i] + S[i+1:] + [j]
                if _is_valid(candidate):
                    if len(candidate) >= len(S):  # Only accept if same or larger size
                        S = candidate
                        S_set = set(S)
                        not_in_S = list(all_elements - S_set)
                        improved = True
                        if len(S) > best_len:
                            best = S[:]
                            best_len = len(S)
                        break
            if improved:
                break
        if not improved:
            break
    
    return best
>>>>>>> REPLACE

