import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 55.0  # Safety margin before 60s kill

# Precompute conflict graph: conflict[x] is a bitset of numbers that conflict with x
# Two numbers conflict if they are part of a 3-AP with some third number
conflict = [0] * (N + 1)

def precompute_conflicts():
    for x in range(1, N + 1):
        # x conflicts with y if there exists z such that {x,y,z} is a 3-AP
        # This happens when y = x+d, z = x+2d OR y = x-d, z = x+d OR y = x-2d, z = x-d
        # Actually, x and y conflict if they can be in a 3-AP together with some z
        # That means: x, y, z form a 3-AP for some z
        # So y must be such that x and y are in the same 3-AP
        # This happens if:
        # - x is first, y is second: z = 2y - x must be in [1,N] and z != x, z != y
        # - x is first, y is third: z = (x+y)/2 must be integer in [1,N]
        # - x is second, y is first: z = 2x - y must be in [1,N]
        # - x is second, y is third: z = 2x - y must be in [1,N]
        # - x is third, y is first: z = (x+y)/2 must be integer
        # - x is third, y is second: z = 2y - x must be in [1,N]
        
        # Simpler: x and y conflict if there exists z != x, z != y such that
        # {x, y, z} is a 3-AP. This means the three values form an arithmetic progression.
        # So sorted(x,y,z) = a, a+d, a+2d for some d >= 1.
        
        # x and y are in a 3-AP iff:
        # 1. y = x + d and x + 2d <= N (x is first, y is second)
        # 2. y = x + 2d and x + d >= 1 (x is first, y is third) - always true if y > x
        # 3. y = x - d and x - 2d >= 1 (x is second, y is first)
        # 4. y = x - 2d (x is second, y is first, but this is same as 3 with different d)
        # Actually let me think more carefully.
        
        # Three distinct numbers a < b < c form a 3-AP if b - a = c - b, i.e., a + c = 2b.
        # So x and y conflict if there exists z such that {x, y, z} = {a, b, c} with a + c = 2b.
        # This means one of x, y, z is the middle term.
        
        # Case 1: x is middle. Then y and z are symmetric around x: z = 2x - y.
        #   Need 1 <= z <= N, z != x, z != y. So z = 2x - y must be in [1,N] and z != x, z != y.
        #   z != x means 2x - y != x => y != x (true since y != x)
        #   z != y means 2x - y != y => y != x (true)
        #   So we just need 1 <= 2x - y <= N.
        
        # Case 2: y is middle. Then x and z are symmetric around y: z = 2y - x.
        #   Need 1 <= z <= N, z != x, z != y.
        
        # Case 3: z is middle. Then x and y are symmetric around z: z = (x + y) / 2.
        #   Need z to be integer, 1 <= z <= N, z != x, z != y.
        #   z integer means (x + y) is even.
        
        conflicts = 0
        for y in range(1, N + 1):
            if y == x:
                continue
            
            # Case 1: x is middle, z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                conflicts |= (1 << (y - 1))
                continue
            
            # Case 2: y is middle, z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                conflicts |= (1 << (y - 1))
                continue
            
            # Case 3: z is middle, z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    conflicts |= (1 << (y - 1))
        
        conflict[x] = conflicts

precompute_conflicts()

def can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset creates a 3-AP.
    Returns True if x can be added without creating a 3-AP.
    """
    # x can be added if no element in bitset conflicts with x
    # But wait - conflict means they CANNOT both be in the set.
    # So x can be added if conflict[x] & bitset == 0
    return (conflict[x] & bitset) == 0

def greedy_construct(order):
    """
    Greedy construction: add elements in the given order if they don't create a 3-AP.
    Returns (bitset, score) where score is the exact sum of reciprocals as a Fraction.
    """
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

# Precompute reciprocal Fractions for speed
recip_frac = [Fraction(0, 1)] * (N + 1)
recip_float = [0.0] * (N + 1)
for i in range(1, N + 1):
    recip_frac[i] = Fraction(1, i)
    recip_float[i] = 1.0 / i

# Precompute suffix sums of reciprocals for bounding
suffix_sums = [0.0] * (N + 2)
for i in range(N, 0, -1):
    suffix_sums[i] = suffix_sums[i + 1] + recip_float[i]

def greedy_color_bound(available, current_bitset):
    """
    Compute a greedy coloring upper bound on the maximum weight independent set
    from the available vertices.
    
    available: bitset of vertices still available to consider
    current_bitset: bitset of vertices already in our independent set
    
    Returns an upper bound on the sum of reciprocals we can still add.
    """
    # Get list of available vertices
    avail_list = []
    temp = available
    while temp:
        # Find lowest set bit
        lsb = temp & (-temp)
        idx = lsb.bit_length()  # This gives position + 1
        v = idx  # vertex number
        avail_list.append(v)
        temp ^= lsb
    
    if not avail_list:
        return 0.0
    
    # Greedy coloring: assign colors greedily to get an upper bound
    # The number of colors used is an upper bound on the size of any independent set
    # But we want a weight-based bound, so we use a weighted version
    
    # For a tighter bound, we can use the fact that in each color class,
    # we can pick at most one vertex. So the upper bound is the sum of the max weight
    # in each color class.
    
    # Simple approach: sort by weight descending, assign colors greedily
    # Upper bound = sum of weights of vertices that get color 1 (first color)
    # Actually, a better bound: for each color class, the max weight in that class
    # contributes to the upper bound.
    
    # Even simpler and often effective: use the fact that the chromatic number
    # gives a bound on independence number. For weighted case, we can use:
    # UB = sum over colors of max weight in that color class
    
    # But computing exact coloring is expensive. Let's use a heuristic:
    # Process vertices in order of decreasing weight, assign smallest available color.
    # Then UB = sum of max weight per color class.
    
    # For speed, let's use a simpler bound: the sum of all weights in available
    # is already a loose bound. A tighter one: use the conflict structure.
    
    # Actually, for maximum weight independent set, a good bound is:
    # For each vertex v in available, if we include v, we can't include any neighbor.
    # A simple bound: sum of top k weights where k is the independence number of the
    # conflict graph on available vertices.
    
    # Let's use a greedy bound: sort available by weight desc, greedily pick
    # non-conflicting vertices, and the sum of picked weights is a LOWER bound.
    # For an UPPER bound, we can use: for each vertex, weight[v] * (1 / (1 + avg_degree))
    # but that's complicated.
    
    # Simple effective bound: the sum of the top min(|available|, alpha) weights,
    # where alpha is an upper bound on the independence number.
    # A simple upper bound on alpha is |available| / (1 + min_degree) by Turan's theorem,
    # but that's loose.
    
    # Let's just use the sum of all available weights as a loose bound for now,
    # and rely on the search to prune effectively.
    
    # Actually, a better approach: use the fact that if we process vertices in
    # a certain order, we can bound how many more we can add.
    
    # For now, return the sum of all available weights (loose but simple)
    total = 0.0
    for v in avail_list:
        total += recip_float[v]
    return total

def branch_and_bound(deadline):
    """
    Branch and Bound search to maximize the reciprocal sum using conflict graph.
    """
    best_score = Fraction(0, 1)
    best_bitset = 0
    best_score_float = 0.0
    
    state = {
        'best_score': best_score,
        'best_bitset': best_bitset,
        'best_score_float': best_score_float,
        'deadline': deadline
    }
    
    # Count nodes visited for debugging
    node_count = [0]
    
    def dfs(current, bitset, score, score_float):
        """
        current: next vertex to consider (1-indexed)
        bitset: current independent set
        score: exact score
        score_float: float score for fast comparison
        """
        # Time check
        if time.time() > state['deadline']:
            return
        
        node_count[0] += 1
        
        # Base case: no more vertices to consider
        if current > N:
            if score > state['best_score']:
                state['best_score'] = score
                state['best_bitset'] = bitset
                state['best_score_float'] = score_float
            return
        
        # Bound: if even adding all remaining can't beat best, prune
        if score_float + suffix_sums[current] <= state['best_score_float'] + 1e-15:
            return
        
        x = current
        
        # Option 1: Include x (if possible)
        if can_add(bitset, x):
            new_bitset = bitset | (1 << (x - 1))
            new_score = score + recip_frac[x]
            new_score_float = score_float + recip_float[x]
            dfs(x + 1, new_bitset, new_score, new_score_float)
            
            # After including x, we can also try to add more vertices that conflict with x
            # but this is handled by the recursion
        
        # Option 2: Exclude x
        dfs(x + 1, bitset, score, score_float)
    
    # Initial greedy to get a good starting bound
    best_initial_score = Fraction(0, 1)
    best_initial_bitset = 0
    best_initial_float = 0.0
    
    # Try sorted order (by reciprocal, i.e., ascending value)
    b, s = greedy_construct(list(range(1, N + 1)))
    s_float = float(s)
    if s > best_initial_score:
        best_initial_score = s
        best_initial_bitset = b
        best_initial_float = s_float
    
    # Try random orders
    for _ in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        b, s = greedy_construct(order)
        s_float = float(s)
        if s > best_initial_score:
            best_initial_score = s
            best_initial_bitset = b
            best_initial_float = s_float
    
    # Try a few more structured orders
    # Even numbers only
    b, s = greedy_construct([x for x in range(1, N + 1) if x % 2 == 0])
    s_float = float(s)
    if s > best_initial_score:
        best_initial_score = s
        best_initial_bitset = b
        best_initial_float = s_float
    
    # Numbers congruent to 1 or 2 mod 3 (avoid 3-APs)
    b, s = greedy_construct([x for x in range(1, N + 1) if x % 3 != 0])
    s_float = float(s)
    if s > best_initial_score:
        best_initial_score = s
        best_initial_bitset = b
        best_initial_float = s_float
    
    state['best_score'] = best_initial_score
    state['best_bitset'] = best_initial_bitset
    state['best_score_float'] = best_initial_float
    
    # Start DFS from x=1
    dfs(1, 0, Fraction(0, 1), 0.0)
    
    return state['best_bitset'], state['best_score']

def local_search(bitset, score, deadline):
    """
    Local search: try to improve by swapping elements.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element that can be added
        for x in range(1, N + 1):
            if not (bitset & (1 << (x - 1))):
                if can_add(bitset, x):
                    bitset |= (1 << (x - 1))
                    score += recip_frac[x]
                    improved = True
        
        # Try removing one element and adding two others
        for x in range(1, N + 1):
            if bitset & (1 << (x - 1)):
                # Try removing x
                new_bitset = bitset & ~(1 << (x - 1))
                new_score = score - recip_frac[x]
                
                # Try adding elements that were previously blocked by x
                added_any = False
                for y in range(1, N + 1):
                    if not (new_bitset & (1 << (y - 1))):
                        if can_add(new_bitset, y):
                            new_bitset |= (1 << (y - 1))
                            new_score += recip_frac[y]
                            added_any = True
                
                # Continue adding more if possible
                more_added = True
                while more_added:
                    more_added = False
                    for y in range(1, N + 1):
                        if not (new_bitset & (1 << (y - 1))):
                            if can_add(new_bitset, y):
                                new_bitset |= (1 << (y - 1))
                                new_score += recip_frac[y]
                                more_added = True
                
                # Check if this improves
                if new_score > score:
                    bitset = new_bitset
                    score = new_score
                    improved = True
    
    return bitset, score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Run Branch and Bound
    best_bitset, best_score = branch_and_bound(deadline)
    
    # Local search polish
    best_bitset, best_score = local_search(best_bitset, best_score, deadline)
    
    # Final polish: try to add any remaining numbers that can be added
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    best_score += recip_frac[x]
                    improved = True
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
