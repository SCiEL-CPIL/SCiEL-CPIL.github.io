# Inside local_search, inside the "if added_count > 1:" block
if added_count > 1:
    # Add addable elements one by one, re-checking constraints
    # because adding one might block another
    for y in addable:
        if can_add_fast(y, current, ap_counts):
            current.add(y)
            for ap_idx in elem_aps[y]:
                ap_counts[ap_idx] += 1
    improved = True
    break
