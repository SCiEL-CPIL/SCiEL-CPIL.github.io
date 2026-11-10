"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _conflicts_with(s, x):
    """Return set of elements in s that form a 4-AP with x."""
    conflicts = set()
    for d in range(1, 513):
        if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
            conflicts.update([x+d, x+2*d, x+3*d])
        if (x - d) in s and (x + d) in s and (x + 2*d) in s:
            conflicts.update([x-d, x+d, x+2*d])
        if (x - 2*d) in s and (x - d) in s and (x + d) in s:
            conflicts.update([x-2*d, x-d, x+d])
        if (x - 3*d) in s and (x - 2*d) in s and (x - d) in s:
            conflicts.update([x-3*d, x-2*d, x-d])
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Greedy construction
    selected = set()
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if not _conflicts_with(selected, x):
            selected.add(x)
    
    if time.time() > deadline:
        return sorted(selected)
    
    # Local search: try to add elements by swapping
    for _pass in range(100):
        if time.time() > deadline:
            break
        improved = False
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in selected:
                continue
            conf = _conflicts_with(selected, x)
            if not conf:
                selected.add(x)
                improved = True
                continue
            # Try removing one conflict element y
            for y in sorted(conf):
                if y not in selected:
                    continue
                # Remove y, add x
                selected.discard(y)
                selected.add(x)
                # Check x doesn't conflict with new set
                if not _conflicts_with(selected, x):
                    improved = True
                    break
                else:
                    # Revert
                    selected.discard(x)
                    selected.add(y)
        if not improved:
            break
    
    return sorted(selected)
# EVOLVE-BLOCK-END
