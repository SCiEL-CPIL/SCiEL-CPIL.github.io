"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations
    
    # Generate all valid pairs and use greedy with local search
    candidates = list(range(2, 257))
    
    # Precompute compatibility: can x and y both be in the set?
    # Two numbers conflict if lcm(x,y) <= 256
    # We want to maximize sum(1/x) over an independent set
    
    # Greedy: process in order of increasing x (higher weight 1/x first)
    # Then try to add if compatible with all already selected
    
    selected = []
    for x in sorted(candidates, key=lambda v: 1.0/v, reverse=True):
        # x sorted by descending 1/x means ascending x
        compatible = True
        for y in selected:
            if lcm(x, y) <= 256:
                compatible = False
                break
        if compatible:
            selected.append(x)
    
    # Local search: try to replace one element with another to improve
    improved = True
    while improved:
        improved = False
        current_sum = sum(1.0/x for x in selected)
        for i, x in enumerate(selected):
            # Try removing x and adding some other y not in selected
            for y in candidates:
                if y in selected:
                    continue
                # Check if y is compatible with all except x
                compatible = True
                for j, z in enumerate(selected):
                    if j == i:
                        continue
                    if lcm(y, z) <= 256:
                        compatible = False
                        break
                if not compatible:
                    continue
                new_sum = current_sum - 1.0/x + 1.0/y
                if new_sum > current_sum + 1e-12:
                    selected[i] = y
                    current_sum = new_sum
                    improved = True
                    break
            if improved:
                break
    
    return selected
# EVOLVE-BLOCK-END
