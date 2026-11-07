<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    n = 256
    
    # Sieve of Eratosthenes for primes up to n
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    
    primes = [i for i in range(2, n + 1) if sieve[i]]
    num_primes = len(primes)
    
    # Precompute pairwise gcd matrix (1 = coprime, 0 = not coprime)
    # For all numbers 2..256
    numbers = list(range(2, n + 1))
    num_numbers = len(numbers)
    coprime = [[False] * num_numbers for _ in range(num_numbers)]
    for i in range(num_numbers):
        for j in range(i + 1, num_numbers):
            coprime[i][j] = coprime[j][i] = (gcd(numbers[i], numbers[j]) == 1)
    
    # Precompute prime factors for each number
    prime_factors = [[] for _ in range(n + 1)]
    for p in primes:
        for j in range(p, n + 1, p):
            prime_factors[j].append(p)
    
    # Greedy initial solution: sort by value descending, include if coprime with all selected
    selected = set()
    for num in sorted(numbers, reverse=True):
        if all(gcd(num, s) == 1 for s in selected):
            selected.add(num)
    
    best_sum = sum(selected) + 1  # +1 for the number 1
    best_solution = list(selected)
    
    # Local search: 1-for-1 and 1-for-2 swaps
    def evaluate_swap_remove_add(current_set, remove, add):
        """Check if removing 'remove' and adding 'add' is valid and returns gain."""
        if add in current_set:
            return 0
        # Check add is coprime with all in current_set except remove
        for x in current_set:
            if x != remove and gcd(add, x) != 1:
                return None
        return add - remove
    
    def evaluate_swap_remove_two_add_one(current_set, remove1, remove2, add):
        """Check if removing remove1, remove2 and adding 'add' is valid and returns gain."""
        if add in current_set:
            return 0
        for x in current_set:
            if x != remove1 and x != remove2 and gcd(add, x) != 1:
                return None
        return add - remove1 - remove2
    
    current_set = set(best_solution)
    current_sum = best_sum - 1  # exclude 1 from sum for calculations
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps
        for rem in list(current_set):
            for add in numbers:
                if add in current_set:
                    continue
                gain = evaluate_swap_remove_add(current_set, rem, add)
                if gain is not None and gain > 0:
                    current_set.discard(rem)
                    current_set.add(add)
                    current_sum += gain
                    if current_sum + 1 > best_sum:
                        best_sum = current_sum + 1
                        best_solution = list(current_set)
                    improved = True
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
        
        # Try 1-for-2 swaps (remove two, add one)
        for rem1 in list(current_set):
            for rem2 in list(current_set):
                if rem1 == rem2:
                    continue
                for add in numbers:
                    if add in current_set:
                        continue
                    gain = evaluate_swap_remove_two_add_one(current_set, rem1, rem2, add)
                    if gain is not None and gain > 0:
                        current_set.discard(rem1)
                        current_set.discard(rem2)
                        current_set.add(add)
                        current_sum += gain
                        if current_sum + 1 > best_sum:
                            best_sum = current_sum + 1
                            best_solution = list(current_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
    
    # Verify all are Python ints
    result = [1] + sorted(best_solution)
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    n = 256
    
    # Sieve of Eratosthenes for primes up to n
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    
    primes = [i for i in range(2, n + 1) if sieve[i]]
    
    # Precompute prime factors for each number 1..256
    prime_factors = [[] for _ in range(n + 1)]
    for p in primes:
        for j in range(p, n + 1, p):
            prime_factors[j].append(p)
    
    # Group numbers by their set of prime factors
    # Two numbers are coprime iff they share no prime factors
    # Build conflict graph: edge between i,j if they share a prime factor
    
    numbers = list(range(2, n + 1))
    num_numbers = len(numbers)
    num_to_idx = {v: i for i, v in enumerate(numbers)}
    
    # Adjacency: for each number, which other numbers conflict with it
    adj = [set() for _ in range(num_numbers)]
    for i in range(num_numbers):
        for p in prime_factors[numbers[i]]:
            for j in range(i + 1, num_numbers):
                if p in prime_factors[numbers[j]]:
                    adj[i].add(j)
                    adj[j].add(i)
    
    # Weights are the values themselves
    weights = [numbers[i] for i in range(num_numbers)]
    
    # Maximum Weight Independent Set via branch-and-bound with coloring bound
    # The complement graph: in conflict graph, we want max weight independent set
    # Equivalently, in the complement (coprime graph), we want max weight clique
    
    # Actually, let's work directly: independent set in conflict graph
    # = set of vertices with no edges between them in conflict graph
    
    # Greedy coloring of the conflict graph for upper bounds
    # For a set of candidate vertices, color them so adjacent vertices get different colors
    # The number of colors is an upper bound on the size of any independent set
    # For weighted case, we use a weighted coloring bound
    
    best_weight = [0]
    best_set = [set()]
    
    def greedy_color_weighted(candidates, adj_local):
        """
        Greedy coloring of candidates in the conflict graph.
        Returns list of (color, total_weight_of_that_color) pairs.
        The sum of weights of the heaviest color per "layer" gives a bound.
        
        Actually for weighted independent set, we use:
        - Order vertices by decreasing weight
        - Assign colors greedily
        - The bound is sum of max weight in each color class
        Wait, that's not right either.
        
        Standard approach for weighted MIS:
        - Use the unweighted coloring bound: if we can color the subgraph with k colors,
          then the independent set has at most k vertices.
        - For weighted: order by weight, and the bound is sum of the first k weights
          where k is the chromatic number bound.
        
        Better: use a weighted greedy bound.
        Process vertices in some order. Maintain a "budget" per color.
        Actually, the standard McCreight-style bound:
        - Color the graph greedily
        - For each color class, take the maximum weight vertex
        - Sum these maxima as an upper bound on the independent set weight
        
        Hmm, that's not quite right. Let me think more carefully.
        
        For unweighted MIS: coloring gives |IS| <= number of colors.
        For weighted: we can use the fact that in each color class (which is a clique in the
        conflict graph... wait no, color classes are independent sets in the conflict graph).
        
        Wait, I'm confusing myself. Let me restart.
        
        In the CONFLICT graph:
        - We want a maximum weight INDEPENDENT SET (no two adjacent in conflict graph)
        - A coloring of the conflict graph partitions vertices into independent sets (color classes)
        - Any clique in the conflict graph has at most one vertex per color class
        - But we want independent sets, not cliques.
        
        Actually, the standard branch-and-bound for MIS uses the coloring of the COMPLEMENT graph.
        Let me reformulate:
        
        In the COPRIME graph (complement of conflict graph):
        - We want a maximum weight CLIQUE
        - A coloring of the coprime graph: color classes are independent sets in coprime graph
          = cliques in the conflict graph
        - Any clique in the coprime graph has at most one vertex per color class of the coprime graph
        - So |clique| <= number of colors
        - For weighted: bound = sum of largest weights, one per color class
        
        This is getting complex. Let me use a simpler but effective approach:
        Branch-and-bound where:
        - We maintain a candidate set C
        - Upper bound: sum of all weights in C (loose but fast)
        - Or better: greedy coloring bound
        
        For the weighted case, a common approach:
        - Order candidates by decreasing weight
        - Greedily assign colors (in the conflict graph)
        - The bound is: for each color, take the max weight vertex in that color class
          (since an independent set can have at most one vertex from each color class... 
           wait no, color classes ARE independent sets, so an independent set can have
           multiple vertices from the same color class)
        
        I'm making an error. Let me be very careful.
        
        In the conflict graph G:
        - Independent set = set of vertices with no edges between them in G
        - Coloring of G: partition into independent sets (color classes)
        - A color class is an independent set, so it CAN be part of our solution
        - The issue is: different color classes may have conflicts between them
        
        The standard trick for MIS (unweighted):
        - If G is k-colorable, then alpha(G) <= k... NO, that's wrong.
        - If G is k-colorable, then the complement is... 
        
        OK let me look at this from the standard algorithm perspective.
        
        For Maximum Independent Set in graph G:
        - Equivalent to Maximum Clique in complement graph G'
        - Branch and bound for Max Clique:
          - Maintain candidate set C (vertices that can join the current clique)
          - Upper bound: |C| (since clique can have at most |C| vertices)
          - Better: color G'[C] with k colors, then clique size <= k
          - Because in G', a color class is an independent set, meaning in G it's a clique... 
            no wait.
        
        Ugh, let me just use a straightforward approach.
        
        For Max Weight Independent Set in conflict graph G:
        - Branch on a vertex v in C:
          - Include v: new C = C ∩ N̄(v) (non-neighbors of v in G), weight += w(v)
          - Exclude v: new C = C \\ {v}
        - Upper bound on C: sum of weights in C (loose)
        - Better bound: greedy coloring of G[C]
          - In G[C], color greedily. Each color class is an independent set in G[C].
          - An independent set in G[C] can have at most one vertex from each... 
            NO. A color class IS an independent set. So an independent set can contain
            ALL vertices of a color class.
          
        I think I've been overcomplicating this. The standard bound for MIS is:
        - Color the graph G[C]
        - The chromatic number chi(G[C]) is a LOWER bound on the clique number of the complement
        - For MIS: alpha(G) = omega(complement of G)
        - So if we color the complement of G[C], the number of colors is an upper bound on omega(complement) = alpha(G[C])
        
        So: to bound the independent set in G[C], we color the COMPLEMENT of G[C] (i.e., the coprime graph restricted to C).
        The number of colors used is an upper bound on the size of the maximum independent set in G[C].
        For weighted: the bound is the sum of the k largest weights, where k is the number of colors.
        
        Let me implement this.
        """
        # Color the coprime subgraph induced by candidates
        # In the coprime graph, edges connect vertices that ARE coprime
        # We want to color this graph
        # Number of colors = upper bound on max independent set in conflict graph = max clique in coprime graph
        
        # Greedy coloring of the coprime graph on candidates
        # Two vertices can share a color if they are NOT adjacent in the coprime graph
        # (i.e., they conflict in the original graph)
        
        # Order candidates by decreasing weight for better bounds
        sorted_cands = sorted(candidates, key=lambda v: -weights[v])
        
        color_assign = {}
        color_weights = []  # list of total weights per color
        color_sets = []  # list of vertex sets per color
        
        for v in sorted_cands:
            # Find which colors v cannot use (colors already used by coprime neighbors)
            used_colors = set()
            for u in color_assign:
                if u in adj[v]:  # u conflicts with v, so u is NOT coprime with v
                    # In the coprime graph, u and v are NOT adjacent
                    # So they CAN share a color
                    pass
                else:
                    # u is coprime with v, so in the coprime graph they ARE adjacent
                    # They cannot share a color
                    used_colors.add(color_assign[u])
            
            # Assign the smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            color_assign[v] = c
            if c >= len(color_weights):
                color_weights.append(0)
                color_sets.append(set())
            color_weights[c] += weights[v]
            color_sets[c].add(v)
        
        # Bound: for weighted case, the maximum weight independent set in G[C]
        # is at most the sum of the top k weights where k = number of colors
        # But a tighter bound: in each color class of the coprime graph, the vertices
        # form a clique in the conflict graph, so at most one can be in the independent set.
        # Wait no - color classes in the coprime graph are independent sets in the coprime graph,
        # which means they are cliques in the conflict graph.
        # So in each color class, at most one vertex can be in our independent set.
        # Therefore, the bound is: for each color class, take the max weight, and sum them.
        
        bound = 0
        for c in range(len(color_weights)):
            max_w = max(weights[v] for v in color_sets[c])
            bound += max_w
        
        return bound, color_assign
    
    # Actually, I realize the coloring approach I described has an issue with efficiency.
    # Let me use a simpler but effective branch-and-bound.
    
    # Simpler approach: use the fact that the problem decomposes by prime.
    # Each number is assigned to at least one prime. If we pick a number, we can't pick
    # any number sharing a prime factor.
    
    # Better decomposition: group by the smallest prime factor.
    # Numbers with smallest prime factor p can only conflict with each other and with
    # numbers that have p as a factor.
    
    # Actually, the cleanest approach for this specific problem:
    # - For each prime p, at most one selected number can be divisible by p
    # - So we're choosing, for each prime, either 0 or 1 number divisible by p
    # - But a number may be divisible by multiple primes, creating coupling
    
    # Let me use a different strategy: branch on primes.
    # For each prime p (in some order), decide which number (if any) to select that is divisible by p.
    # A number divisible by p and q constrains both p and q.
    
    # This is essentially a weighted matching/set packing problem.
    
    # Given the small size (54 primes up to 256, 255 numbers), let me use a 
    # branch-and-bound on the primes.
    
    # Primes up to 256
    all_primes = [p for p in range(2, n + 1) if sieve[p]]
    
    # For each prime, list of numbers divisible by it (sorted by value desc)
    prime_to_numbers = {}
    for p in all_primes:
        prime_to_numbers[p] = [x for x in numbers if x % p == 0]
    
    # For each number, list of primes it's divisible by
    num_to_primes = {}
    for x in numbers:
        num_to_primes[x] = prime_factors[x]
    
    # Branch and bound on primes
    # State: which primes are "used", which numbers are selected
    # Order primes by number of available options (most constrained first)
    
    # Precompute: for each prime, the numbers that have it as a factor
    # and are not yet excluded
    
    # Let me use a recursive B&B with memoization-like pruning
    
    best_total = [0]
    best_selection = [set()]
    
    def solve_bnb(prime_idx, used_primes, selected, current_sum):
        """
        Branch and bound over primes.
        prime_idx: index into ordered_primes
        used_primes: set of primes that are already "consumed" (have a selected number)
        selected: set of selected numbers
        current_sum: sum of selected numbers so far
        """
        if time.time() > deadline:
            return
        
        if prime_idx >= len(ordered_primes):
            if current_sum > best_total[0]:
                best_total[0] = current_sum
                best_selection[0] = set(selected)
            return
        
        p = ordered_primes[prime_idx]
        
        # Upper bound: current_sum + sum of max available for each remaining prime
        # For each remaining prime, the best we can do is the largest number not yet excluded
        ub = current_sum
        for j in range(prime_idx, len(ordered_primes)):
            pj = ordered_primes[j]
            if pj in used_primes:
                continue
            # Find the best number for pj that doesn't conflict with selected
            best_for_pj = 0
            for x in prime_to_numbers[pj]:
                if x in selected:
                    continue
                # Check if x conflicts with any selected number
                conflict = False
                for sp in num_to_primes[x]:
                    if sp in used_primes and sp != pj:
                        # x uses prime sp which is already used by another number
                        # Actually, we need to check if x shares a prime with any selected number
                        conflict = True
                        break
                if not conflict:
                    # Also check x doesn't share primes with selected numbers
                    ok = True
                    for sel in selected:
                        if any(sp in num_to_primes[sel] for sp in num_to_primes[x]):
                            ok = False
                            break
                    if ok and x > best_for_pj:
                        best_for_pj = x
            ub += best_for_pj
        
        if ub <= best_total[0]:
            return
        
        # Option 1: skip this prime (don't select any number with factor p)
        solve_bnb(prime_idx + 1, used_primes, selected, current_sum)
        
        # Option 2: select a number divisible by p
        # Try numbers in descending order of value
        for x in prime_to_numbers[p]:
            if x in selected:
                continue
            # Check x is coprime with all selected
            ok = True
            for sel in selected:
                if gcd(x, sel) != 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Select x
            new_used = used_primes | set(num_to_primes[x])
            new_selected = selected | {x}
            solve_bnb(prime_idx + 1, new_used, new_selected, current_sum + x)
    
    # Order primes by number of candidate numbers (ascending = most constrained first)
    # But also consider: primes with larger numbers should be branched earlier?
    # Let's order by: (number of candidates, -max_candidate_value)
    ordered_primes = sorted(all_primes, key=lambda p: (len(prime_to_numbers[p]), -max(prime_to_numbers[p])))
    
    # This B&B might be too slow. Let me add a greedy initial solution and 
    # use it as a lower bound.
    
    # Greedy: sort numbers by value desc, add if coprime with all selected
    greedy_selected = set()
    for num in sorted(numbers, reverse=True):
        if all(gcd(num, s) == 1 for s in greedy_selected):
            greedy_selected.add(num)
    
    best_total[0] = sum(greedy_selected)
    best_selection[0] = set(greedy_selected)
    
    # Run B&B
    solve_bnb(0, set(), set(), 0)
    
    # Local search to improve
    current_set = set(best_selection[0])
    current_sum = sum(current_set)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try 1-for-1 swaps
        for rem in list(current_set):
            for add in numbers:
                if add in current_set:
                    continue
                gain = add - rem
                if gain <= 0:
                    continue
                # Check add is coprime with all in current_set except rem
                ok = True
                for x in current_set:
                    if x != rem and gcd(add, x) != 1:
                        ok = False
                        break
                if ok:
                    current_set.discard(rem)
                    current_set.add(add)
                    current_sum += gain
                    if current_sum > best_total[0]:
                        best_total[0] = current_sum
                        best_selection[0] = set(current_set)
                    improved = True
                    break
            if improved:
                break
    
    # Add 1 (it's coprime with everything)
    result = sorted(best_selection[0]) + [1]
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize my B&B approach has issues - the upper bound calculation is O(n^2) per node which will be too slow, and the branching factor is large. Let me reconsider and use a cleaner, more efficient approach that stays within the "branch-and-bound with coloring" family as specified.

