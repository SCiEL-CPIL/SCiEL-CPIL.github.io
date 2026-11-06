import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # Safety margin before 90s kill
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = gcd(i + 1, j + 1)

    # Identify bad triples
    # A triple (i, j, k) is bad if gcd(i,j) == gcd(i,k) == gcd(j,k)
    # We want to find a maximum subset with NO bad triples.
    # This is equivalent to finding a Maximum Independent Set (MIS) in a 3-uniform hypergraph
    # where vertices are [0, N) and edges are the bad triples.
    # However, standard MIS solvers work on graphs (2-uniform).
    # We can transform this: A set is independent in the hypergraph if it contains no edge.
    # This is a 3-SAT-like problem or a specific type of constraint satisfaction.
    # 
    # Alternative view: 
    # If we select a set S, it is valid iff for every bad triple (i,j,k), S does not contain all three.
    # This is equivalent to finding a Maximum Independent Set in a graph G' constructed as follows?
    # No, that's not a direct graph transformation unless we expand the state space.
    #
    # However, we can use a Branch and Bound approach directly on the hypergraph.
    # State: (remaining_vertices, current_selected)
    # Branch on a vertex v:
    #   1. Include v: Remove v from remaining. Also, we must ensure no bad triple is completed.
    #      This is tricky because including v doesn't immediately remove other vertices, 
    #      but it creates a constraint: for any bad triple (v, a, b), we cannot select BOTH a and b.
    #      This turns the problem into a graph MIS problem dynamically? 
    #      Actually, if we select v, the remaining problem is:
    #      Find max independent set in the induced subgraph/hypergraph on remaining vertices,
    #      subject to constraints that for each edge (v, a, b) in the hypergraph, {a, b} is a forbidden pair.
    #      This means the "conflict graph" changes.
    #
    # Simpler Approach: Branch and Bound on the Hypergraph directly.
    # Heuristic: 
    # 1. Greedy Lower Bound: Build a large valid set quickly.
    # 2. Upper Bound: Use a fractional cover or simple counting. 
    #    A better upper bound for 3-uniform hypergraph independent set is hard.
    #    Let's use a simpler strategy: 
    #    Convert to a standard Graph MIS problem by a reduction?
    #    No, the constraint "not all 3" is weaker than "not any 2".
    #    
    #    Let's stick to a specialized Branch and Bound for the 3-uniform hypergraph.
    #    Branching variable: The vertex involved in the most remaining bad triples?
    #    
    #    Actually, there is a known transformation.
    #    If we want to avoid {i,j,k}, it is equivalent to saying that in the selected set, 
    #    the neighborhood of any vertex is an independent set in the graph of its "partners"?
    #    
    #    Let's try a different structural insight.
    #    Most bad triples involve small gcds (like 1 or 2).
    #    Numbers that are powers of 2: gcd(2^a, 2^b) = 2^min(a,b).
    #    Triple (2^a, 2^b, 2^c) with a<b<c: gcds are 2^a, 2^a, 2^b. Not equal.
    #    So powers of 2 are always safe? 
    #    gcd(2^a, 2^b) = 2^a, gcd(2^a, 2^c) = 2^a, gcd(2^b, 2^c) = 2^b.
    #    Equal only if a=b, which is impossible for distinct elements.
    #    So all powers of 2 can be included.
    #    What about primes? 
    #    gcd(p, q) = 1 for distinct primes.
    #    Triple of distinct primes: gcds are 1, 1, 1. EQUAL.
    #    So we cannot have 3 distinct primes!
    #    This is a huge structural constraint.
    #    
    #    Let's verify: 
    #    If we pick 3 distinct primes p, q, r.
    #    gcd(p,q)=1, gcd(p,r)=1, gcd(q,r)=1.
    #    They are equal. So {p,q,r} is a bad triple.
    #    Therefore, the set can contain at most 2 distinct primes.
    #    
    #    What about prime powers?
    #    p^a, q^b, r^c with distinct primes p,q,r.
    #    gcds are 1,1,1. Bad.
    #    So at most 2 numbers with distinct prime factors?
    #    No, "distinct primes" means the set of prime factors is disjoint?
    #    If x, y, z are pairwise coprime, then gcds are 1. Bad.
    #    So we cannot have 3 pairwise coprime numbers.
    #    
    #    This suggests a strategy based on prime factor overlap.
    #    If we group numbers by their smallest prime factor?
    #    Or simply: 
    #    The condition "gcd(x,y)=gcd(x,z)=gcd(y,z)=d" implies that x=d*a, y=d*b, z=d*c
    #    where gcd(a,b)=gcd(a,c)=gcd(b,c)=1.
    #    So a, b, c are pairwise coprime.
    #    Thus, a bad triple exists if there are 3 numbers in the set which, after dividing by their common gcd d, become pairwise coprime.
    #    
    #    This is a very specific structure.
    #    Let's use a Branch and Bound solver optimized for this.
    #    
    #    Since N=128 is small, we can use a bitmask for the set.
    #    Precompute bad_triples as a list of bitmasks? Or indices.
    #    For speed, let's use a list of bad triples (i,j,k).
    
    bad_triples = []
    for i in range(N):
        for j in range(i + 1, N):
            g_ij = gcd_mat[i][j]
            for k in range(j + 1, N):
                if gcd_mat[i][k] == g_ij and gcd_mat[j][k] == g_ij:
                    bad_triples.append((i, j, k))
    
    # Map element to triples
    elem_to_triples = [[] for _ in range(N)]
    for t in bad_triples:
        for idx in t:
            elem_to_triples[idx].append(t)
    
    # Greedy Lower Bound
    def greedy_lower_bound():
        # Start with all, remove worst offender
        mask = (1 << N) - 1
        # Sort elements by degree (number of triples) descending?
        # Or just iterate and remove if needed?
        # Let's do a simple greedy: 
        # Order elements randomly? No, deterministic is safer for reproducibility if needed, 
        # but random helps escape local minima.
        # Let's try a few random orders.
        best_size = 0
        best_mask = 0
        
        for _ in range(20):
            if time.time() > DEADLINE - 20: break
            order = list(range(N))
            random.shuffle(order)
            current_mask = 0
            for idx in order:
                # Try to add idx
                new_mask = current_mask | (1 << idx)
                # Check if adding idx creates any bad triple
                # Only need to check triples containing idx where other 2 are in current_mask
                valid = True
                for t in elem_to_triples[idx]:
                    if (new_mask >> t[0]) & 1 and (new_mask >> t[1]) & 1 and (new_mask >> t[2]) & 1:
                        valid = False
                        break
                if valid:
                    current_mask = new_mask
            
            size = bin(current_mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = current_mask
                
        return best_mask, best_size
    
    init_mask, init_size = greedy_lower_bound()
    best_size = init_size
    best_mask = init_mask
    
    # Branch and Bound
    # State: (available_mask, current_mask)
    # available_mask: bits that can still be added
    # current_mask: bits already selected
    
    # Upper Bound Heuristic:
    # For each vertex in available, count how many triples it is in with other available vertices?
    # A simple bound: 1 + max_independent_set(available without neighbors)?
    # For hypergraph, a simple bound is hard.
    # Let's use: size_of_current + number_of_available
    # This is weak.
    # Better: 
    # If we pick a vertex v, we don't remove others immediately, but we add constraints.
    # This makes B&B complex.
    
    # Alternative: Use a SAT solver? No, standard library only.
    # Let's use a simple recursive B&B with a strong pruning rule:
    # If the number of available vertices + current size <= best_size, prune.
    # This is weak but safe.
    
    # To make it stronger, we can use a greedy packing bound on the remaining vertices.
    # Greedy pack the remaining vertices into a valid set (ignoring current_mask constraints? No, must respect them).
    # Actually, we can just run a fast greedy on the available set to get a lower bound for the subproblem.
    # If current_size + greedy_bound(available) <= best_size, prune.
    
    def fast_greedy_bound(avail_mask):
        # Greedily select from avail_mask
        # Simple: pick vertices with lowest conflict degree in the induced subgraph?
        # Or just random?
        # For speed, let's just count bits.
        # Actually, a simple heuristic: 
        # The max independent set in a 3-uniform hypergraph with m vertices is at least m / (something)?
        # Let's just use the count of bits as a very weak bound, 
        # but rely on the fact that N=128 is small enough for a well-ordered search?
        # 2^128 is too big.
        
        # Let's try a different approach: 
        # The problem is equivalent to finding a Maximum Independent Set in a graph?
        # No.
        
        # Let's use a simple local search from the greedy result as the main engine,
        # but with a smarter initialization and more aggressive swaps.
        # The previous program did this. Why did it score 1.0?
        # Maybe it didn't find the optimum.
        # The inspirations mention CSP and Structural construction.
        
        # Let's try the Structural Construction mentioned in Inspiration 2.
        # "Primes, prime powers, and carefully selected composites"
        # We established: Max 2 pairwise coprime numbers.
        # This implies that for any prime p, we can pick numbers that share p.
        # If we pick all multiples of 2:
        # 2, 4, 6, 8...
        # gcd(2,4)=2, gcd(2,6)=2, gcd(4,6)=2.
        # Triple (2,4,6): gcds 2,2,2. BAD.
        # So we can't just take all multiples of 2.
        
        # What if we take numbers with distinct "shapes"?
        
        # Let's go back to the B&B idea but with a better bound.
        # Bound: For the remaining available vertices, what is the max number we can add?
        # We can compute a greedy independent set on the available vertices (ignoring current_mask)
        # and use that as an upper bound? No, that's a lower bound for the subproblem.
        # We need an upper bound.
        # A valid upper bound is simply the number of available vertices.
        # To improve, we can use a fractional relaxation?
        
        # Given the time limit, let's implement a robust Local Search with:
        # 1. Strong Greedy Initialization (multiple random orders)
        # 2. 1-1 Swaps
        # 3. 2-2 Swaps (remove 2, add 2)
        # 4. Simulated Annealing style acceptance?
        
        pass

    # Let's implement the improved Local Search.
    
    def is_valid_mask(mask):
        for t in bad_triples:
            if (mask >> t[0]) & 1 and (mask >> t[1]) & 1 and (mask >> t[2]) & 1:
                return False
        return True

    def get_conflict_count(mask, idx):
        count = 0
        for t in elem_to_triples[idx]:
            if (mask >> t[0]) & 1 and (mask >> t[1]) & 1 and (mask >> t[2]) & 1:
                count += 1
        return count

    def local_search(start_mask):
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        improved = True
        
        # Phase 1: Make valid
        while not is_valid_mask(current_mask):
            max_conf = -1
            candidates = []
            for i in range(N):
                if (current_mask >> i) & 1:
                    c = get_conflict_count(current_mask, i)
                    if c > max_conf:
                        max_conf = c
                        candidates = [i]
                    elif c == max_conf and c > 0:
                        candidates.append(i)
            if not candidates: break
            to_remove = random.choice(candidates)
            current_mask &= ~(1 << to_remove)
            current_size -= 1
            if time.time() > DEADLINE - 5:
                return current_mask, current_size

        # Phase 2: Improve
        while improved and time.time() < DEADLINE - 2:
            improved = False
            
            # 1. Try single add
            missing = [i for i in range(N) if not ((current_mask >> i) & 1)]
            random.shuffle(missing)
            for i in missing:
                new_mask = current_mask | (1 << i)
                if is_valid_mask(new_mask):
                    current_mask = new_mask
                    current_size += 1
                    improved = True
                    break
            
            # 2. Try 1-1 swap
            if not improved:
                present = [i for i in range(N) if (current_mask >> i) & 1]
                missing = [i for i in range(N) if not ((current_mask >> i) & 1)]
                random.shuffle(present)
                random.shuffle(missing)
                
                # Limit swaps to check for speed
                for r in present[:10]:
                    temp_mask = current_mask & ~(1 << r)
                    for a in missing[:10]:
                        if a == r: continue
                        new_mask = temp_mask | (1 << a)
                        if is_valid_mask(new_mask):
                            # Check if this swap allows further additions?
                            # For now, accept if it doesn't decrease size (it doesn't)
                            # But we only accept if it leads to a better state?
                            # A swap keeps size same. We accept it to escape local minima.
                            # We should only accept if it's "different" from current?
                            # It is.
                            current_mask = new_mask
                            improved = True
                            break
                    if improved: break

            # 3. Try 2-2 swap
            if not improved:
                present = [i for i in range(N) if (current_mask >> i) & 1]
                missing = [i for i in range(N) if not ((current_mask >> i) & 1)]
                
                # Sample a few pairs
                if len(present) >= 2 and len(missing) >= 2:
                    r1, r2 = random.sample(present, 2)
                    a1, a2 = random.sample(missing, 2)
                    
                    temp_mask = current_mask
                    temp_mask &= ~(1 << r1)
                    temp_mask &= ~(1 << r2)
                    temp_mask |= (1 << a1)
                    temp_mask |= (1 << a2)
                    
                    if is_valid_mask(temp_mask):
                        # Accept? 
                        # Again, size is same. 
                        # We accept to explore.
                        current_mask = temp_mask
                        improved = True

        return current_mask, current_size

    # Run multiple restarts
    # 1. Start from greedy best
    m, s = local_search(init_mask)
    if s > best_size:
        best_size = s
        best_mask = m

    # 2. Random restarts
    for _ in range(100):
        if time.time() > DEADLINE - 10: break
        
        # Random start
        size = random.randint(N // 2, N)
        subset = random.sample(range(N), size)
        start_mask = 0
        for i in subset:
            start_mask |= (1 << i)
        
        m, s = local_search(start_mask)
        if s > best_size:
            best_size = s
            best_mask = m

    # Convert to list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)
