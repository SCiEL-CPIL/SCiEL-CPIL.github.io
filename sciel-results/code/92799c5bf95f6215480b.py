"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    selected = []
    
    for p in primes:
        # Get all powers of p <= 40
        powers = []
        exp = 1
        val = p
        while val <= 40:
            powers.append((exp, val))
            exp += 1
            val *= p
            
        best_subset = []
        
        # Try to find the largest subset of powers with distinct subset sums
        # Max size is len(powers)
        for size in range(len(powers), 0, -1):
            found = False
            for comb in itertools.combinations(powers, size):
                exps = [e for e, v in comb]
                # Check distinct subset sums
                sums = set()
                valid = True
                for r in range(len(exps) + 1):
                    for sub in itertools.combinations(exps, r):
                        s = sum(sub)
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    best_subset = [v for e, v in comb]
                    found = True
                    break
            if found:
                break
        
        selected.extend(best_subset)

    return selected
# EVOLVE-BLOCK-END
