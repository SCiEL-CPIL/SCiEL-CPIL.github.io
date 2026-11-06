parents_raw = dag.get('parents', {})
if isinstance(parents_raw, dict):
    parents = {k: list(v) for k, v in parents_raw.items()}
elif isinstance(parents_raw, (list, tuple)):
    # Try to interpret as list of parent lists (index = vertex)
    # Or list of pairs
    # Check first element
    if len(parents_raw) > 0:
        first = parents_raw[0]
        if isinstance(first, (list, tuple)) and len(first) == 2 and not isinstance(first[0], (list, tuple, set)):
             # Likely list of pairs (vertex, parents)
             # But wait, if first is (1, [2]), first[0] is 1 (int), first[1] is [2] (list).
             # If first is [1, 2], it's ambiguous.
             # Let's assume if it's a list of lists, it's index-based.
             # If it's a list of tuples/pairs, it's pair-based.
             # The error "length 0" strongly points to list of lists where one list is empty?
             # No, dict() iterates. If element is [], len is 0.
             # So if parents_raw = [[], [1]], dict() fails on first element [].
             # So it is likely a list of lists.
             pass
    # Heuristic: If elements are lists/tuples of length != 2, or if they look like parent lists, assume index-based.
    # If elements are pairs (x, y) where y is a list, assume pair-based.
    
    # Let's try to detect.
    is_pair_list = False
    if len(parents_raw) > 0:
        # Check if all elements are length 2 and second element is a list/tuple/set
        for item in parents_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                if isinstance(item[1], (list, tuple, set)):
                    is_pair_list = True
                    break
                else:
                    is_pair_list = False
                    break
            else:
                is_pair_list = False
                break
                
    if is_pair_list:
        parents = {item[0]: list(item[1]) for item in parents_raw}
    else:
        # Assume index-based: parents_raw[i] is parents of vertex i
        parents = {i: list(p) for i, p in enumerate(parents_raw)}
else:
    parents = {}
