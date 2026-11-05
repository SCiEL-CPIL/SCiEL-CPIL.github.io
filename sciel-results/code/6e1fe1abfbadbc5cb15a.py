import time
import random
from math import gcd
from itertools import combinations

def construct():
    """
    Strategy:
    1. Construct a large candidate set S0 using a structural rule:
       - Include 1.
       - For each prime p, include exactly one element of the form p^k (highest k <= 128).
       - Include all numbers that are products of two distinct primes (semiprimes) 
         where the primes are "large" or chosen to minimize gcd overlaps, 
         specifically: include x if x is square-free and has exactly 2 prime factors, 
         OR x is a prime power (already handled), OR x is 1.
       Actually, a simpler structural rule that often yields large valid sets:
       Include all numbers with an odd number of prime factors (with multiplicity)? 
       No, let's use a known heuristic: 
       The condition "no three distinct a,b,c with gcd(a,b)=gcd(a,c)=gcd(b,c)" is equivalent to 
       saying that in the graph where edges are labeled by gcd, there are no triangles with same label.
       
       A strong structural candidate:
       Let S be the set of all numbers in [1, 128] that are NOT divisible by any small prime square? 
       
       Let's try a different structural approach:
       Partition [1, 128] into "classes" based on their radical or prime support.
       
       Actually, the most robust structural approach for this specific constraint:
       Consider the set of all numbers that are powers of 2? No, too small.
       
       Let's use a "randomized structural" seed:
       Generate multiple random subsets of size ~60-70, then repair.
       But the directive says "different design family". 
       
       Let's try: **Minimum Hitting Set Repair**.
       1. Start with a set S0 that is likely to have few conflicts.
          A good S0: All numbers in [1, 128] that are odd? 
          Let's check: If a,b,c are odd, gcds are odd. 
          Does {1, 3, 9} have gcd(1,3)=1, gcd(1,9)=1, gcd(3,9)=3. Not equal.
          {3, 9, 27}: gcd(3,9)=3, gcd(3,27)=3, gcd(9,27)=9. Not equal.
          {9, 27, 81}: gcd(9,27)=9, gcd(9,81)=9, gcd(27,81)=27. Not equal.
          It seems powers of 3 don't form bad triples.
          
          Let's try S0 = all odd numbers in [1, 128]. Size = 64.
          Check for bad triples in odds.
          If we find bad triples, we remove the minimum number of elements to hit all of them.
          
    2. Identify all bad triples in S0.
    3. Solve a Set Cover problem (greedy approximation) to remove elements that cover the most bad triples.
    4. Iterate until no bad triples remain.
    5. Try to add back elements from [1,128]\S if they fit.
    """
    N = 128
    deadline = time.time() + 85.0

    # Precompute gcd table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g

    def find_bad_triples(S):
        """Find all triples in S that are bad."""
        bad_triples = []
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                gab = gt[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gab == gt[a][c] == gt[b][c]:
                        bad_triples.append((a, b, c))
        return bad_triples

    def greedy_set_cover(universe, sets):
        """
        universe: list of items to cover
        sets: list of sets, each set is a subset of universe
        Returns the indices of the sets chosen (greedy).
        """
        if not universe:
            return []
        covered = set()
        chosen = []
        remaining_universe = set(universe)
        
        # For efficiency, map each item to the sets that contain it? 
        # Or just iterate. Since N is small, simple greedy is fine.
        # sets[i] is the set of triples that element i (from the domain of removal candidates) covers.
        # Actually, let's define the problem properly.
        # We want to remove elements from S to break all bad_triples.
        # Universe = bad_triples
        # Each element x in S "covers" the set of triples T such that x in T.
        
        # Build coverage map
        coverage = {}
        for x in S:
            coverage[x] = set()
        for t in universe:
            for x in t:
                coverage[x].add(t)
        
        while remaining_universe:
            best_x = None
            best_count = 0
            for x in S:
                # Count how many uncovered triples x covers
                count = 0
                for t in coverage[x]:
                    if t in remaining_universe:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_x = x
            
            if best_count == 0:
                # Should not happen if universe is non-empty and coverage is correct
                break
            
            chosen.append(best_x)
            remaining_universe -= coverage[best_x]
            
        return chosen

    def repair(S):
        """
        Repair S by removing minimum elements to eliminate all bad triples.
        """
        S = set(S)
        while time.time() < deadline:
            bad_triples = find_bad_triples(S)
            if not bad_triples:
                break
            
            # Greedy set cover to remove elements
            removals = greedy_set_cover(bad_triples, S)
            if not removals:
                break
            
            # Remove the first element from the greedy cover (or all? Greedy returns a set of elements to remove)
            # The greedy_set_cover returns a list of elements to remove to cover all triples.
            # Let's remove them all.
            for x in removals:
                if x in S:
                    S.discard(x)
        
        return S

    def try_add_back(S):
        """Try to add any element from [1, 128] not in S."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S:
                    new_S = S | {x}
                    # Check if adding x creates any bad triple
                    # Only need to check triples involving x
                    valid = True
                    S_list = sorted(S)
                    for i in range(len(S_list)):
                        a = S_list[i]
                        gax = gt[x][a]
                        for j in range(i + 1, len(S_list)):
                            b = S_list[j]
                            if gax == gt[x][b] == gt[a][b]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S.add(x)
                        changed = True
        return S

    best_set = set()
    best_size = 0

    # Structural seeds
    # Seed 1: All odd numbers
    S1 = set(range(1, N + 1, 2))
    # Seed 2: All numbers not divisible by 4 (i.e., v2(x) <= 1)
    S2 = set(x for x in range(1, N + 1) if x % 4 != 0)
    # Seed 3: All numbers with odd number of prime factors (with multiplicity)
    def omega(n):
        c = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    S3 = set(x for x in range(1, N + 1) if omega(x) % 2 == 1)
    # Seed 4: Random subsets of size 70
    random.seed(42)
    for _ in range(5):
        S_rand = set(random.sample(range(1, N + 1), 70))
        S_rand.add(1) # 1 is usually safe
        # Repair and add back
        S_repaired = repair(S_rand)
        S_final = try_add_back(S_repaired)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)

    # Process structural seeds
    for S_seed in [S1, S2, S3]:
        if time.time() > deadline:
            break
        S_repaired = repair(S_seed)
        S_final = try_add_back(S_repaired)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)

    # Final polish: try to add more
    if time.time() < deadline:
        best_set = try_add_back(best_set)

    if not best_set:
        best_set = {1}

    return sorted(best_set)
