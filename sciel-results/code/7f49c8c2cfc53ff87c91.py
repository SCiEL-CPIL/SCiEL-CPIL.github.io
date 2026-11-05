import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflict graph: edge (u,v) if (u*v) % (u+v) == 0
    # Use the characterization: let g=gcd(u,v), u'=u/g, v'=v/g, conflict iff (u'+v') | g
    conflicts = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = 0
            a, b = i, j
            while b:
                a, b = b, a % b
            g = a
            u_prime = i // g
            v_prime = j // g
            if g % (u_prime + v_prime) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    def can_add(v, s):
        """Check if vertex v can be added to set s."""
        for u in s:
            if v in conflicts[u]:
                return False
        return True
    
    # Greedy construction
    best_set = set()
    
    def greedy_by_order(order):
        result = set()
        for v in order:
            if can_add(v, result):
                result.add(v)
        return result
    
    # Try multiple orderings
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    order_odd_first = sorted(vertices, key=lambda x: (x % 2 == 0, degrees[x]))
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))
    
    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        set(order_odd),
        set(order_even),
        greedy_by_order(order_odd_first),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
    ]
    
    # Multiple random shuffles
    for _ in range(30):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))
    
    for c in candidates:
        if len(c) > len(best_set):
            best_set = set(c)
    
    # Local search: add vertices, then try 1-for-2 swaps
    def local_search(s, time_limit):
        s = set(s)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Phase 1: Try adding each excluded vertex
            excluded = [v for v in vertices if v not in s]
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s):
                    s.add(v)
                    improved = True
            
            # Phase 2: Try 1-for-2 swaps (remove one, add two non-conflicting)
            if time.time() < time_limit:
                s_list = list(s)
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    temp = s - {u}
                    # Find all vertices that can be added to temp
                    addable = []
                    for v in vertices:
                        if v not in s and v != u and can_add(v, temp):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 15)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if v1 not in conflicts[v2]:
                                    s = temp | {v1, v2}
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
        return s
    
    # Run local search from best candidate
    time_limit = deadline - 15.0
    best_set = local_search(best_set, time_limit)
    
    # Multiple restarts with random perturbations
    while time.time() < deadline - 15.0:
        perturbed = set(best_set)
        if len(perturbed) > 1:
            to_remove = random.choice(list(perturbed))
            perturbed.discard(to_remove)
        
        result = local_search(perturbed, deadline - 15.0)
        if len(result) > len(best_set):
            best_set = result
    
    # Final validation
    result_list = sorted(best_set)
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2))
    
    return result_list
# EVOLVE-BLOCK-END
