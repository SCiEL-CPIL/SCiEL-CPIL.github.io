import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        index_to_val = {v: i for i, v in enumerate(values)}

        # Build conflict graph: edge between i and j if values[i] + values[j] in v_set
        # This is a 2-uniform graph approximation of the 3-uniform hypergraph problem.
        # But wait, the constraint is a+b != c for all a,b,c in S.
        # If we just avoid pairs (i,j) where v_i + v_j in S, that's not enough because the sum could be a third element.
        # Actually, if v_i + v_j = v_k, then we have a triangle (i,j,k) in the "sum graph".
        # If we select i and j, we must NOT select k.
        # This is exactly a 3-uniform hypergraph independent set.
        # However, a common heuristic is to treat it as a graph where an edge exists between i and j if v_i + v_j is in the set.
        # If we find a Max Independent Set in this graph, does it guarantee sum-freeness?
        # No. Consider a=1, b=2, c=3. 1+2=3.
        # Graph edges: (1,2) because 1+2=3 in set. (1,3) because 1+3=4 (not in set? maybe). (2,3) because 2+3=5 (not in set?).
        # If the set is {1,2,3}, the graph has edge (1,2).
        # Max independent set of {1,2,3} graph: can pick {1,3} or {2,3}.
        # {1,3}: 1+3=4 not in {1,3}. OK. But wait, is 1+1=2 in {1,3}? No.
        # So {1,3} is sum-free.
        # What if set is {1, 2, 3, 4}?
        # 1+3=4. Edge (1,3).
        # 2+2=4. Edge (2,2)? No, simple graph.
        # 1+1=2. Edge (1,1)?
        # The constraint a+b=c includes a=b.
        # So if 2 is in S, 4 cannot be in S (if 2+2=4).
        # Graph approach: Edge between u and v if u+v is in V (the full instance set).
        # If we pick an independent set I in this graph:
        # For any a,b in I, a+b is NOT in I?
        # If a+b = c where c is in I, then there is an edge between a and b (since c in V).
        # So a and b cannot both be in I.
        # Thus, an independent set in the graph "G=(V, E) where {i,j} in E iff v_i+v_j in V" IS sum-free.
        # Proof:
        # Suppose S is an independent set in G.
        # Assume for contradiction there exist a,b,c in S such that a+b=c.
        # Then c is in V (since S subset V).
        # So v_a + v_b = v_c, which means edge (a,b) exists in G.
        # But S is independent, so a and b cannot both be in S. Contradiction.
        # Therefore, Max Independent Set of G is a valid sum-free subset.
        # And any sum-free subset is an independent set in G.
        # So the problem is EXACTLY Max Independent Set in G!

        # Build adjacency list
        adj = [set() for _ in range(n)]
        # For all pairs (i, j) with i < j
        # If values[i] + values[j] in v_set, add edge.
        # Also need to handle a=a case?
        # If a+a=c, i.e., 2a=c.
        # Then edge between (a, a)? No, graph is simple.
        # But the condition "a+b=c with a=b allowed" means if 2a=c, we cannot have a and c in S.
        # In the graph G, is there an edge between a and c?
        # Edge exists if a+c in V? No, edge exists if a+c in V?
        # My definition: Edge (i,j) iff v_i + v_j in V.
        # If 2a = c, then v_a + v_a = v_c.
        # This implies v_a + v_a is in V.
        # But the edge definition is v_i + v_j.
        # If i != j, we check v_i + v_j.
        # If i = j, we check v_i + v_i.
        # Standard graph MIS doesn't handle loops.
        # If we have a loop at a (meaning 2a in V), then we cannot select a?
        # No, we can select a, but we cannot select c.
        # The constraint is: if a in S, then c cannot be in S.
        # This is a constraint between a and c.
        # So we should add an edge between a and c if 2a = c.
        # Let's re-define the graph construction carefully.
        
        # Constraint: For all a, b in S, a+b not in S.
        # This is equivalent to: For all a, b in V, if a+b=c in V, then not (a in S and b in S and c in S).
        # This is a 3-uniform hypergraph independent set.
        
        # Is it equivalent to MIS on a 2-graph?
        # Let's check the counter-example.
        # V = {1, 2, 3}.
        # Triples: (1,1,2), (1,2,3), (2,1,3), (1,3,4-no), (2,2,4-no).
        # Hyperedges: {1,2}, {1,2,3} -> wait, {1,1,2} means if 1 in S, 2 not in S.
        # {1,2,3} means if 1,2 in S, 3 not in S.
        
        # Graph G edges:
        # (1,2) because 1+2=3 in V.
        # (1,1) loop? 1+1=2 in V.
        # (2,2) loop? 2+2=4 not in V.
        
        # If we ignore loops, G has edge (1,2).
        # MIS of G: {1,3} or {2,3}.
        # {1,3}: 1+3=4 not in S. 1+1=2 not in S. 3+3=6 not in S. Sum-free.
        # {2,3}: 2+3=5 not in S. 2+2=4 not in S. Sum-free.
        
        # What if V = {1, 2, 4}?
        # 1+1=2. Constraint: not(1 in S and 2 in S).
        # 2+2=4. Constraint: not(2 in S and 4 in S).
        # 1+2=3 (not in V).
        # 1+4=5 (not in V).
        
        # Graph edges (i != j):
        # (1,2): 1+2=3 not in V. No edge.
        # (1,4): 1+4=5 not in V. No edge.
        # (2,4): 2+4=6 not in V. No edge.
        
        # Graph G is empty. MIS is {1,2,4}.
        # But {1,2,4} is NOT sum-free because 1+1=2.
        # So the 2-graph MIS approach FAILS if we ignore self-sums (loops).
        
        # How to handle loops?
        # If 2a = c, we have a constraint between a and c.
        # So we should add an edge between a and c.
        # Let's redefine:
        # Edge between i and j (i != j) if:
        # 1. v_i + v_j in V  (standard sum)
        # 2. v_i + v_i = v_j (i.e., 2*v_i = v_j) -> then edge (i, j)
        # 3. v_j + v_j = v_i (i.e., 2*v_j = v_i) -> then edge (j, i) i.e. (i, j)
        
        # Let's check V={1,2,4} again with this new rule.
        # i=1 (val 1), j=2 (val 2): 2*1=2. Edge (1,2).
        # i=2 (val 2), j=4 (val 4): 2*2=4. Edge (2,4).
        # i=1 (val 1), j=4 (val 4): 2*1!=4, 2*4!=1, 1+4!=1. No edge.
        
        # Graph G edges: (1,2), (2,4).
        # MIS: {1,4} (size 2) or {2} (size 1) or {1,4} is independent?
        # 1 and 4 are not connected. Yes.
        # Is {1,4} sum-free?
        # 1+1=2 not in {1,4}.
        # 1+4=5 not in {1,4}.
        # 4+4=8 not in {1,4}.
        # Yes!
        
        # So the graph G with edges:
        # {i,j} in E iff i != j and (v_i + v_j in V OR 2*v_i == v_j OR 2*v_j == v_i)
        # Wait, is this sufficient?
        # If we pick an independent set S in this G:
        # For any a, b in S:
        # If a+b=c in V:
        # If a != b, then (a,b) is an edge in G (by rule 1). So a,b not both in S.
        # If a == b, then 2a=c. Then (a,c) is an edge in G (by rule 2/3).
        # If a in S and c in S, they are adjacent, contradiction.
        # So S is sum-free.
        
        # Is it possible that S is sum-free but not an independent set in G?
        # If (a,b) in E, then either v_a+v_b in V or 2v_a=v_b or 2v_b=v_a.
        # If v_a+v_b=v_c in V, and a,b,c in S, then a+b=c, violation.
        # If 2v_a=v_b, and a,b in S, then a+a=b, violation.
        # So any independent set in G is sum-free.
        
        # Thus, Max Sum-Free Subset = Max Independent Set in this specific graph G.

        adj = [set() for _ in range(n)]
        
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                # Check if edge exists
                if (vi + vj) in v_set:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * vi == vj:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * vj == vi:
                    adj[i].add(j)
                    adj[j].add(i)

        # Now solve Max Independent Set on G
        # Greedy + SA
        
        best_S_idx = set()
        best_size = 0

        # Greedy initialization strategies
        # 1. Random order
        # 2. Degree ascending (pick low degree first? or high?)
        #    For MIS, picking high degree first is usually bad (removes many neighbors).
        #    Picking low degree first might be better?
        #    Standard greedy for MIS: Pick vertex with min degree, add to IS, remove it and its neighbors.
        #    Or: Pick vertex with max "something"?
        #    Let's try Min Degree Greedy.
        
        def greedy_mis(min_degree=True):
            rem = set(range(n))
            is_set = set()
            # Copy degrees
            degs = [len(adj[i]) for i in range(n)]
            # Dynamic degrees
            while rem:
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Find min degree vertex in rem
                min_v = None
                min_d = float('inf')
                for v in rem:
                    if degs[v] < min_d:
                        min_d = degs[v]
                        min_v = v
                if min_v is None:
                    break
                is_set.add(min_v)
                rem.discard(min_v)
                for u in adj[min_v]:
                    if u in rem:
                        rem.discard(u)
                        for w in adj[u]:
                            if w in rem:
                                degs[w] -= 1
                        degs[u] = 0 # just mark as removed
            return is_set

        def greedy_mis_max_first():
            rem = set(range(n))
            is_set = set()
            degs = [len(adj[i]) for i in range(n)]
            while rem:
                if time.time() - start_time > DEADLINE - 10:
                    break
                max_v = None
                max_d = -1
                for v in rem:
                    if degs[v] > max_d:
                        max_d = degs[v]
                        max_v = v
                if max_v is None:
                    break
                is_set.add(max_v)
                rem.discard(max_v)
                for u in adj[max_v]:
                    if u in rem:
                        rem.discard(u)
                        for w in adj[u]:
                            if w in rem:
                                degs[w] -= 1
                        degs[u] = 0
            return is_set

        # Run multiple greedy restarts
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 20:
                break
            # Randomize the order slightly or use different tie-breaking
            # Simplest: just run min-degree greedy, it's deterministic for same input.
            # To get variety, we can randomly remove some vertices first?
            # Or just run SA from this start.
            
            start_set = greedy_mis()
            if len(start_set) > best_size:
                best_size = len(start_set)
                best_S_idx = set(start_set)

        # Simulated Annealing on the graph
        # State: set of indices S
        # Move: add or remove a random vertex
        # Energy: -len(S)  (we want to maximize size)
        # Constraint: S must be independent.
        # If we allow invalid states, we need a penalty.
        # Penalty: number of edges within S.
        # Score = len(S) - lambda * edges_in_S
        
        S_idx = set(best_S_idx)
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        lambda_penalty = 100  # High penalty for violations
        
        def get_edges_in_S(S):
            count = 0
            for u in S:
                for v in adj[u]:
                    if v in S:
                        count += 1
            return count // 2  # Each edge counted twice
        
        current_edges = get_edges_in_S(S_idx)
        current_score = len(S_idx) - lambda_penalty * current_edges
        
        while T > T_min and time.time() - start_time < DEADLINE - 5:
            # Pick a random vertex
            v = random.choice(range(n))
            
            if v in S_idx:
                new_S = S_idx - {v}
                # Remove edges incident to v that are within S
                # edges_removed = number of neighbors of v in S_idx
                edges_removed = 0
                for u in adj[v]:
                    if u in S_idx:
                        edges_removed += 1
                new_edges = current_edges - edges_removed
                new_score = len(new_S) - lambda_penalty * new_edges
            else:
                new_S = S_idx | {v}
                # Add edges incident to v that are within S_idx
                edges_added = 0
                for u in adj[v]:
                    if u in S_idx:
                        edges_added += 1
                new_edges = current_edges + edges_added
                new_score = len(new_S) - lambda_penalty * new_edges
            
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                S_idx = new_S
                current_edges = new_edges
                current_score = new_score
                
                # If we have a valid solution (0 edges), update best
                if current_edges == 0 and len(S_idx) > best_size:
                    best_size = len(S_idx)
                    best_S_idx = set(S_idx)
            
            T *= cooling

        # Convert indices back to values
        best_values = [values[i] for i in best_S_idx]
        return best_values

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            vals = instance(size, seed)
            # Fallback: All odds
            res = [x for x in vals if x % 2 != 0]
            if not is_sum_free(res):
                # If odds fail (e.g. 1+1=2, but 2 is even so not in res? Wait.
                # If S is all odd numbers, a+b is even.
                # If no even numbers in S, then a+b is not in S.
                # So All Odds is ALWAYS sum-free.
                res = [x for x in vals if x != 0] # Just in case
            results.append(res)
            continue

        # Allocate budget
        # 60: ~20s, 120: ~40s, 240: ~60s?
        # Total 120s.
        budget = min(remaining - 5, 10 + 50 * (size / 240))
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to all odds
            res = [x for x in vals if x % 2 != 0]
        
        results.append(res)

    return results
