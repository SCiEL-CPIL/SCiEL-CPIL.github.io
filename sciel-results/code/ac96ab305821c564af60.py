"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict_graph(values):
    """
    Builds adjacency list for the conflict graph.
    Edge (i, j) exists if values[i] + values[j] is in values.
    Note: a+b=c implies a,b conflict. If a=b, 2a=c, so a conflicts with itself?
    The problem says "a=b allowed". So if 2a=c and a,c in set, it's invalid.
    In the graph model for MIS:
    We want to select a set S such that no a,b,c in S satisfy a+b=c.
    This is NOT exactly MIS on the graph where edge=(i,j) if sum in set.
    If we pick i and j, and sum is k, we must not pick k.
    This is a 3-uniform hypergraph independent set problem, or equivalently, 
    we can view it as: if we pick i, we cannot pick j if i+j is in S? No.
    
    Let's re-read carefully: "no a + b = c with a, b, c in it".
    This means the set S must not contain any solution to x+y=z.
    
    Standard reduction to MIS:
    Create a graph G where vertices are the elements.
    Draw an edge between i and j if values[i] + values[j] is in the set of ALL values.
    If we select an independent set in this graph, does it guarantee sum-free?
    If i and j are in the independent set, there is no edge between them.
    This means values[i] + values[j] is NOT in the set of ALL values.
    Therefore, values[i] + values[j] cannot be in the subset S (since S is a subset of ALL values).
    So, if i, j in S, then i+j is not in ALL values, so i+j is not in S.
    Thus, no a,b,c in S satisfy a+b=c.
    
    Wait, what if a=b?
    If a=b, then 2a=c.
    In the graph, is there a self-loop or edge?
    The graph definition usually considers distinct vertices.
    If we have an edge (i,j) for i!=j, and we pick i and j, we ensure i+j not in S.
    What about 2i=c?
    If we pick i, we must ensure c is not in S.
    This is not captured by a simple edge between i and another vertex unless we add a self-loop logic or handle it separately.
    
    However, the standard "Sum-Free" problem reduction to MIS often involves:
    1. If a+b=c, then we cannot have {a,b,c} all in S.
    This is a 3-clique constraint in the complement? No.
    
    Let's look at the current program's logic:
    `adj[i].add(j)` if `values[i] + values[j] in val_set`.
    If we find a Maximum Independent Set (MIS) of this graph, we ensure that for any pair i,j in the set, i+j is NOT in the *entire* universe of values.
    Consequently, i+j is not in the subset.
    So a+b=c is impossible for a!=b.
    
    What about a=b?
    If a=a, 2a=c.
    The graph edge logic above doesn't prevent picking a and c if 2a=c, because the edge is between a and a? No, the loop `for j in range(i+1, n)` skips i==j.
    So if 2a=c, there is no edge between a and a (self loop not stored) nor necessarily between a and c (unless a+c is in set).
    So a simple MIS on the pairwise sum graph might fail for the a=b case.
    
    Example: S = {1, 2}. 1+1=2.
    Graph: Edge(0,1) because 1+2=3? No.
    Edge(0,0)? Skipped.
    Is 1+1 in val_set? Yes (2).
    But the code `for j in range(i+1, n)` misses the case where i==j.
    
    We need to handle the `2a=c` case.
    If we pick `a`, we cannot pick `c` if `2a=c`.
    This is equivalent to adding an edge between `a` and `c` in the conflict graph?
    If we add an edge between `a` and `c` whenever `2a=c`, then an independent set cannot contain both `a` and `c`.
    This correctly handles the `a=b` case.
    
    So, the conflict graph should have an edge between i and j (i != j) if:
    1. values[i] + values[j] is in val_set (covers a!=b)
    2. 2*values[i] == values[j] OR 2*values[j] == values[i] (covers a=b, specifically preventing picking the base and the double)
    
    Let's verify:
    If we have edge (i,j) because 2*values[i] == values[j], we cannot pick both.
    If we pick i, we don't pick j. So 2*values[i] is not in S.
    If we pick j, we don't pick i.
    Does this prevent a+b=c where a=b?
    Yes, because if a=b, then c=2a. We prevent picking a and c together.
    
    So the graph construction should be:
    For all pairs i < j:
      if values[i] + values[j] in val_set: add edge
      if 2*values[i] == values[j]: add edge
      if 2*values[j] == values[i]: add edge
    """
    val_set = set(values)
    n = len(values)
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            vj = values[j]
            # Case 1: vi + vj = vk
            if vi + vj in val_set:
                adj[i].add(j)
                adj[j].add(i)
            # Case 2: 2*vi = vj
            if 2 * vi == vj:
                adj[i].add(j)
                adj[j].add(i)
            # Case 3: 2*vj = vi
            elif 2 * vj == vi:
                adj[i].add(j)
                adj[j].add(i)
                
    return adj

def _is_sum_free(values, subset_indices):
    """Check if the subset is sum-free."""
    sub_set = set(values[i] for i in subset_indices)
    for i in subset_indices:
        for j in subset_indices:
            if values[i] + values[j] in sub_set:
                return False
    return True

def _greedy_mis(adj, n, rng, order=None):
    """Greedy Maximum Independent Set."""
    if order is None:
        order = list(range(n))
        rng.shuffle(order)
        
    in_set = [False] * n
    remaining = set(range(n))
    
    for v in order:
        if v not in remaining:
            continue
        in_set[v] = True
        remaining.discard(v)
        for nb in adj[v]:
            if nb in remaining:
                remaining.discard(nb)
                
    return in_set

def _greedy_abs(adj, n, values):
    """Greedy by decreasing absolute value."""
    order = sorted(range(n), key=lambda i: -abs(values[i]))
    in_set = [False] * n
    remaining = set(range(n))
    
    for v in order:
        if v not in remaining:
            continue
        in_set[v] = True
        remaining.discard(v)
        for nb in adj[v]:
            if nb in remaining:
                remaining.discard(nb)
                
    return in_set

def _simulated_annealing(values, adj, n, rng, start_mask, time_budget):
    """
    Simulated Annealing on the conflict graph.
    State: bitmask of selected vertices.
    Move: flip a random bit.
    Objective: maximize size, penalize conflicts.
    """
    # Convert start_mask to list of indices for easy size counting? No, bitmask is faster.
    # We need to calculate conflicts quickly.
    # conflicts(mask) = number of edges (i,j) such that both i and j are in mask.
    
    # Pre-compute adj_bitmasks for faster AND operations
    adj_masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_masks[i] = mask
        
    def get_conflicts(mask):
        cnt = 0
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            i = lsb.bit_length() - 1
            # Count neighbors of i that are also in mask
            # adj_masks[i] & mask gives neighbors in mask
            # We count each edge twice (once for i, once for j), so divide by 2?
            # Or just sum and divide by 2 at the end.
            cnt += (adj_masks[i] & mask).bit_count()
            m ^= lsb
        return cnt // 2

    def get_size(mask):
        return mask.bit_count()

    cur_mask = start_mask
    cur_conf = get_conflicts(cur_mask)
    cur_size = get_size(cur_mask)
    
    # Objective: size - penalty * conflicts
    # If conflicts > 0, it's invalid. We want to reach 0 conflicts.
    # To escape local optima, we allow invalid states.
    def objective(mask):
        s = mask.bit_count()
        c = get_conflicts(mask)
        if c > 0:
            return s - 1000 * c
        return s

    cur_obj = objective(cur_mask)
    best_mask = cur_mask
    best_obj = cur_obj
    
    t_start = 50.0
    t_end = 0.01
    start_time = time.time()
    
    # Estimate iterations per second to set max_iters based on time
    # We'll just loop until time runs out.
    while time.time() - start_time < time_budget:
        # Decay temperature
        # Simple linear or exponential decay. Let's use exponential.
        # We need to know total time to calculate decay factor per step?
        # Better: just use a fixed number of steps and map to time?
        # Or: t = t_start * (t_end/t_start) ** (elapsed / budget)
        elapsed = time.time() - start_time
        if time_budget <= 0: break
        progress = min(1.0, elapsed / time_budget)
        t = t_start * (t_end / t_start) ** progress
        
        # Random move: flip a random bit
        v = rng.randint(0, n - 1)
        new_mask = cur_mask ^ (1 << v)
        
        new_obj = objective(new_mask)
        delta = new_obj - cur_obj
        
        if delta >= 0 or rng.random() < math.exp(delta / max(t, 1e-10)):
            cur_mask = new_mask
            cur_obj = new_obj
            
            # Check if valid and better
            if (cur_mask & ~best_mask) != 0: # Potential improvement
                conf = get_conflicts(cur_mask)
                if conf == 0:
                    size = cur_mask.bit_count()
                    best_size = best_mask.bit_count()
                    if size > best_size:
                        best_mask = cur_mask
                        best_obj = cur_obj
        else:
            # Revert
            pass # cur_mask unchanged
            
    return best_mask

def _solve_instance(values, time_budget, seed):
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    adj = _build_conflict_graph(values)
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Parity (Odds)
    odds_mask = 0
    for i in range(n):
        if values[i] % 2 != 0:
            odds_mask |= (1 << i)
    if _is_sum_free(values, [i for i in range(n) if odds_mask & (1<<i)]):
        size = odds_mask.bit_count()
        if size > best_size:
            best_size = size
            best_mask = odds_mask
            
    # Strategy 2: Greedy MIS (Random Order)
    for _ in range(5):
        greedy = _greedy_mis(adj, n, rng)
        mask = 0
        for i in range(n):
            if greedy[i]:
                mask |= (1 << i)
        size = mask.bit_count()
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Strategy 3: Greedy Abs Value
    greedy_abs = _greedy_abs(adj, n, values)
    mask = 0
    for i in range(n):
        if greedy_abs[i]:
            mask |= (1 << i)
    size = mask.bit_count()
    if size > best_size:
        best_size = size
        best_mask = mask
        
    # Strategy 4: Simulated Annealing from best start
    if time_budget > 0.5:
        # Use remaining time for SA
        sa_time = time_budget
        sa_mask = _simulated_annealing(values, adj, n, rng, best_mask, sa_time)
        size = sa_mask.bit_count()
        # SA should only return valid masks if we enforce it in update logic, 
        # but let's double check validity just in case of bugs in conflict calc
        # Actually, _simulated_annealing only updates best if conf==0.
        if size > best_size:
            best_size = size
            best_mask = sa_mask

    # Convert mask to list of values
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(values[i])
            
    return result

_start_time = None

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 115  # 5s safety margin
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    total_weight = sum(len(v)**2 for v in instances)
    results = []
    
    for i, values in enumerate(instances):
        elapsed = time.time() - _start_time
        remaining = deadline - elapsed
        if remaining < 1.0:
            remaining = 0.1
            
        weight = (len(values)**2) / total_weight
        time_budget = remaining * weight
        
        subset = _solve_instance(values, time_budget, 801 + i)
        results.append(subset)
        
    return results
# EVOLVE-BLOCK-END
