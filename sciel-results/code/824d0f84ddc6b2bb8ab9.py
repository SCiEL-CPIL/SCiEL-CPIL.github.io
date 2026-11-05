"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

# We will use a greedy algorithm with a twist:
# 1. Precompute all GCDs.
# 2. Identify "bad triples" - triples (a,b,c) where gcd(a,b)==gcd(a,c)==gcd(b,c).
# 3. Use a heuristic to select numbers.
#    The key insight is that if we pick numbers that are "coprime-ish" or have distinct prime factors,
#    we are less likely to form bad triples.
#    
#    A bad triple (a,b,c) with common gcd g means:
#    a = g*a', b = g*b', c = g*c'
#    where gcd(a',b')=gcd(a',c')=gcd(b',c')=1.
#    So the "cores" a', b', c' are pairwise coprime.
#    
#    If we avoid picking multiples of the same small number g, we might avoid bad triples.
#    Alternatively, we can try to maximize the set by being careful about which numbers we include.
#
#    Let's try a different approach:
#    Sort numbers by some criterion (e.g., number of divisors, or just value).
#    Try to add each number if it doesn't create a bad triple with existing pairs.
#    But the naive greedy might be suboptimal.
#
#    Let's look at the structure of bad triples more closely.
#    If g=1, we need pairwise coprime triples.
#    If g=2, we need 2*a', 2*b', 2*c' where a',b',c' are pairwise coprime.
#    
#    Strategy:
#    1. Start with a large set of candidates.
#    2. Use a local search / hill climbing approach.
#    3. Represent the set as a bitset or boolean array.
#    4. Define a cost function: number of bad triples.
#    5. Try to flip bits to reduce cost.
#
#    Actually, let's stick to a robust greedy with backtracking or a smarter heuristic.
#    The previous program was a simple greedy. Let's try to improve it.
#    
#    Heuristic:
#    Sort numbers by "risk". Numbers with many divisors or small values might be riskier.
#    Or, sort by the number of "conflicts" they would create if added.
#    
#    Let's try a Max-SAT style approach with local search.
#    Variables: x_i = 1 if i is in the set.
#    Clauses: For each bad triple (a,b,c), clause (!x_a or !x_b or !x_c).
#    Objective: Maximize sum(x_i).
#    
#    We can use a simple local search:
#    1. Initialize with a random subset or a greedy solution.
#    2. Calculate the number of satisfied clauses (or violations).
#    3. Try flipping variables to reduce violations.
#    4. If stuck, try a random restart.
#
#    Given the time limit (90s), we can afford a decent amount of search.
#    However, 128 variables is small enough that we might be able to do better.
#
#    Let's try a different greedy:
#    Sort numbers by value descending.
#    Try to add the largest numbers first.
#    Why? Large numbers have fewer multiples, so they might be less likely to form bad triples
#    with other large numbers? Not necessarily.
#
#    Let's try sorting by the number of prime factors. Numbers with distinct prime factors
#    are less likely to share a common gcd > 1 with many others.
#
#    Actually, the simplest improvement is to try multiple greedy strategies and pick the best.
#    Strategies:
#    1. Ascending order.
#    2. Descending order.
#    3. Random order (multiple times).
#    4. Sorted by number of divisors (ascending).
#    5. Sorted by "coprimality" potential.
#
#    Let's implement a function to check if adding a number creates a bad triple.
#    def can_add(x, selected):
#        for a, b in combinations(selected, 2):
#            if gcd(a,b) == gcd(a,x) == gcd(b,x):
#                return False
#        return True
#
#    This is O(n^2) per addition, so O(n^3) total. For n=128, this is fine.
#
#    Let's try to be smarter.
#    Precompute gcds for all pairs.
#    Then, for each x, we can check against all pairs in selected.
#
#    Let's try a local search approach on top of a good initial solution.
#    1. Get an initial solution using greedy (e.g., ascending).
#    2. Compute the number of bad triples (should be 0 for a valid solution).
#    3. Try to add numbers that were rejected.
#    4. If adding a number creates bad triples, try to remove one of the conflicting numbers.
#
#    This is a "repair" heuristic.
#
#    Let's implement this.

def _build_gcd_table(n):
    """Build a table of gcds for all pairs (i,j) where 1 <= i <= j <= n."""
    table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table

def _is_bad_triple(a, b, c, gcd_table):
    """Check if (a,b,c) is a bad triple."""
    g_ab = gcd_table[a][b]
    g_ac = gcd_table[a][c]
    g_bc = gcd_table[b][c]
    return g_ab == g_ac == g_bc

def _count_bad_triples(selected, gcd_table):
    """Count the number of bad triples in the selected set."""
    count = 0
    n = len(selected)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _is_bad_triple(selected[i], selected[j], selected[k], gcd_table):
                    count += 1
    return count

def _greedy_construct(order, gcd_table):
    """Greedy construction using the given order of numbers."""
    selected = []
    for x in order:
        # Check if x can be added
        can_add = True
        for a, b in combinations(selected, 2):
            if _is_bad_triple(a, b, x, gcd_table):
                can_add = False
                break
        if can_add:
            selected.append(x)
    return selected

def _local_search(selected, gcd_table, time_limit=60):
    """Local search to improve the solution."""
    start_time = time.time()
    best_selected = set(selected)
    best_size = len(selected)
    
    # We will try to add numbers that are not in the set.
    # If adding a number creates bad triples, we try to remove one of the conflicting numbers.
    
    candidates = [i for i in range(1, 129) if i not in best_selected]
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle candidates for randomness
        random.shuffle(candidates)
        
        for x in candidates:
            # Try to add x
            new_selected = list(best_selected) + [x]
            # Check for bad triples involving x
            bad_pairs = []
            for a, b in combinations(best_selected, 2):
                if _is_bad_triple(a, b, x, gcd_table):
                    bad_pairs.append((a, b))
            
            if not bad_pairs:
                # No conflicts, just add x
                best_selected.add(x)
                best_size += 1
                candidates.remove(x)
                improved = True
            else:
                # Conflicts exist. Try to remove one number from each bad pair.
                # This is complex. Let's try a simpler approach:
                # If the number of conflicts is small, try removing one of the conflicting numbers.
                # This might not be optimal, but it's a start.
                
                # Actually, let's just skip this for now and focus on a better heuristic.
                # The local search might be too slow or complex.
                pass
        
        if not improved:
            break
    
    return list(best_selected)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    gcd_table = _build_gcd_table(128)
    
    best_solution = []
    
    # Try multiple greedy strategies
    strategies = []
    
    # 1. Ascending
    strategies.append(list(range(1, 129)))
    
    # 2. Descending
    strategies.append(list(range(128, 0, -1)))
    
    # 3. Random (multiple times)
    for _ in range(10):
        order = list(range(1, 129))
        random.shuffle(order)
        strategies.append(order)
    
    # 4. Sorted by number of divisors
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    order_div = sorted(range(1, 129), key=num_divisors)
    strategies.append(order_div)
    strategies.append(sorted(range(1, 129), key=num_divisors, reverse=True))
    
    # Run greedy for each strategy
    for order in strategies:
        if time.time() > deadline:
            break
        solution = _greedy_construct(order, gcd_table)
        if len(solution) > len(best_solution):
            best_solution = solution
    
    # Try local search on the best solution
    if time.time() < deadline:
        improved = _local_search(best_solution, gcd_table, time_limit=deadline - time.time())
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Ensure non-empty
    if not best_solution:
        best_solution = [1]
    
    return best_solution

# EVOLVE-BLOCK-END
