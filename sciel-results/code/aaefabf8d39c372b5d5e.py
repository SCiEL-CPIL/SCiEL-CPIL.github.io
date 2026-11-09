"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations
    
    def is_bad_triple(a, b, c):
        return lcm(a, b) == lcm(a, c) == lcm(b, c)
    
    # Constructive approach: 
    # 1. Take all powers of 2: {1, 2, 4, 8, 16, 32, 64, 128}
    # 2. Add primes > 2 that don't create bad triples
    # 3. Add composite numbers carefully
    
    # Actually, let's try: all numbers that are odd OR powers of 2
    # Check if this works
    
    candidate = set()
    for x in range(1, 129):
        # x is power of 2?
        if x & (x - 1) == 0:
            candidate.add(x)
        elif x % 2 == 1:
            candidate.add(x)
    
    # Verify
    for a, b, c in combinations(sorted(candidate), 3):
        if is_bad_triple(a, b, c):
            # Found a bad triple, need to fix
            pass
    
    # Let's just do a greedy from a good starting set
    # Start with all odd numbers + powers of 2, then remove minimal elements to fix conflicts
    
    all_odds = [x for x in range(1, 129) if x % 2 == 1]
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
    initial = set(all_odds + powers_of_2)
    
    # Find all bad triples within initial
    bads = []
    for a, b, c in combinations(sorted(initial), 3):
        if is_bad_triple(a, b, c):
            bads.append((a, b, c))
    
    # Greedily remove elements to break all bad triples
    remaining = set(initial)
    while True:
        # Find a bad triple in remaining
        found = None
        for a, b, c in combinations(sorted(remaining), 3):
            if is_bad_triple(a, b, c):
                found = (a, b, c)
                break
        if found is None:
            break
        # Remove the element that appears in the most bad triples (or just remove one)
        a, b, c = found
        # Remove the one with smallest value to keep larger numbers? Or remove strategically
        # Let's remove the one that's in the most remaining bad triples
        counts = {a: 0, b: 0, c: 0}
        for x, y, z in combinations(sorted(remaining), 3):
            if is_bad_triple(x, y, z):
                for v in (x, y, z):
                    if v in counts:
                        counts[v] += 1
        to_remove = max(counts, key=counts.get)
        remaining.remove(to_remove)
    
    return sorted(remaining)
# EVOLVE-BLOCK-END
