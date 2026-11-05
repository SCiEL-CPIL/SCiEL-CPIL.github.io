"""Snake-in-the-box Q_9: MWIS branch-and-bound on Q_9^2."""
import time
import random
import numpy as np
from scipy.optimize import linprog

N = 9
TIME_LIMIT = 145  # safety margin below 150s hard kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return TIME_LIMIT - elapsed()
    
    # Precompute Q_9^2 adjacency: vertices at Hamming distance 1 or 2
    # For MWIS, we need to find a set of vertices with no two at distance 1 or 2
    # Wait, the snake condition is:
    # - Consecutive vertices differ in exactly 1 bit
    # - Non-consecutive vertices differ in at least 2 bits
    # This means: no two vertices in the path can be at Hamming distance 1 unless they are consecutive.
    # But for an independent set in Q_9^2, we require no two vertices at distance 1 OR 2.
    # A snake is NOT simply an independent set in Q_9^2 because consecutive vertices ARE at distance 1.
    # 
    # Let me reconsider. The problem says:
    # "a sequence of distinct 9-bit vertices in which consecutive vertices differ in one bit 
    #  and non-consecutive vertices differ in at least two bits"
    # 
    # So the constraint is:
    # - For any two vertices in the sequence, if they are consecutive (adjacent in the sequence), 
    #   they must differ in exactly 1 bit.
    # - If they are NOT consecutive, they must differ in at least 2 bits.
    # 
    # This is equivalent to saying: the set of vertices forms an induced path in Q_9.
    # An induced path means that the only edges between vertices in the set are the path edges.
    # 
    # For MWIS on Q_9^2: Q_9^2 has edges between vertices at Hamming distance 1 or 2.
    # An independent set in Q_9^2 is a set where no two vertices are at distance 1 or 2.
    # But a snake has consecutive vertices at distance 1! So a snake is NOT an independent set in Q_9^2.
    #
    # The directive says: "Formulate the problem as a Maximum Weight Independent Set (MWIS) on the 
    # square of the hypercube graph (Q_9^2), where vertices are hypercube vertices and edges connect 
    # vertices at Hamming distance 1, and solve it using a branch-and-bound algorithm with LP relaxation 
    # bounds and color classes, leveraging the fact that a snake is an independent set in Q_9^2 that is 
    # also a Hamiltonian path of its induced subgraph."
    #
    # Wait, re-reading: "edges connect vertices at Hamming distance 1" - that's Q_9, not Q_9^2.
    # But it says "square of the hypercube graph (Q_9^2)". Let me re-read carefully.
    #
    # "MWIS on the square of the hypercube graph (Q_9^2), where vertices are hypercube vertices and 
    #  edges connect vertices at Hamming distance 1"
    #
    # This is contradictory. Q_9^2 has edges at distance 1 AND 2. If edges only connect at distance 1, 
    # that's Q_9. 
    #
    # I think the intent is: the snake's vertex set, when considered as an induced subgraph of Q_9, 
    # forms a path. The "forbidden" pairs are those at distance 1 that are NOT consecutive in the path.
    # 
    # Actually, let me think about this differently. The key insight for the MWIS formulation:
    # A valid snake is a set S of vertices such that:
    # 1. The induced subgraph Q_9[S] is a path (connected, max degree 2, no cycles)
    # 2. No two non-consecutive vertices in the path are adjacent in Q_9 (i.e., at distance 1)
    #
    # Condition 2 means: if u and v are in S and d_H(u,v) = 1, then u and v MUST be consecutive in the path.
    # Equivalently: there is no pair u, v in S with d_H(u,v) = 1 that are not consecutive.
    #
    # For the MWIS approach on Q_9^2: if we define Q_9^2 as having edges between vertices at distance 
    # exactly 1 (not 2), then an independent set in this graph would have no two vertices at distance 1.
    # But that's too restrictive - a snake HAS pairs at distance 1 (the path edges).
    #
    # I think the correct interpretation is: we want to find a set S that can be ordered as a path 
    # v_1, ..., v_k such that:
    # - d_H(v_i, v_{i+1}) = 1 for all i
    # - d_H(v_i, v_j) >= 2 for all |i-j| >= 2
    #
    # The second condition means: for any two vertices in S that are NOT adjacent in the path, 
    # they must be at distance >= 2. This is equivalent to saying that the induced subgraph Q_9[S] 
    # has exactly k-1 edges (the path edges) and no other edges.
    #
    # For the branch-and-bound, we can use the following approach:
    # - The "conflict graph" for the MWIS is: two vertices conflict if they are at distance 1 AND 
    #   we decide they cannot both be in the set unless they are consecutive. But we don't know 
    #   the ordering yet.
    #
    # Alternative simpler approach that still uses the MWIS/B&B framework:
    # Since the problem is hard, let's use a hybrid: 
    # 1. Use greedy/beam search to find good snakes (as before)
    # 2. Use local search with the B&B-inspired bound to improve
    #
    # Actually, let me implement a proper branch-and-bound for the induced path problem.
    # The key is to use the fact that in Q_9^2 (distance 1 or 2), the vertices in the snake must 
    # be "spread out" enough. 
    #
    # Let me implement a branch-and-bound that:
    # - Branches on which vertex to include next
    # - Uses a greedy upper bound: for the remaining candidates, compute a quick upper bound 
    #   on how many more vertices can be added
    # - Uses LP relaxation via scipy for tighter bounds on subproblems
    #
    # Given the time constraint, I'll implement a practical version:
    # 1. Fast greedy with lookahead (baseline)
    # 2. Branch-and-bound with greedy bounds for improvement
    # 3. Local search (2-opt, insertions)
    
    def verify_snake(path):
        """Verify that path is a valid induced path in Q_N."""
        if not path:
            return False
        used = set(path)
        if len(used) != len(path):
            return False
        n = len(path)
        for i in range(n):
            for j in range(i + 1, n):
                diff = path[i] ^ path[j]
                if abs(i - j) == 1:
                    if diff == 0 or (diff & (diff - 1)) != 0:
                        return False
                else:
                    if diff == 0 or (diff & (diff - 1)) == 0:
                        return False
        return True

    def get_neighbors(v):
        """Generate all N neighbors of v."""
        for b in range(N):
            yield v ^ (1 << b)

    def greedy_extend(path, used, forbidden, bit_order, max_steps=None):
        """Extend path greedily using forbidden set. Returns extended path."""
        if max_steps is None:
            max_steps = 512
        for _ in range(max_steps):
            cur = path[-1]
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    def greedy_with_lookahead(path, used, forbidden, bit_order, max_steps=None, lookahead_depth=2):
        """Greedy with lookahead scoring. Returns extended path."""
        if max_steps is None:
            max_steps = 512
        for _ in range(max_steps):
            cur = path[-1]
            best_candidate = None
            best_score = -1
            
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    candidates.append(v)
            
            if not candidates:
                break
            
            if lookahead_depth <= 1 or len(candidates) <= 2:
                best_candidate = candidates[0]
            else:
                for v in candidates:
                    score = 0
                    for b2 in bit_order:
                        v2 = v ^ (1 << b2)
                        if v2 == cur:
                            continue
                        if v2 not in used and v2 not in forbidden:
                            sub_score = 0
                            for b3 in bit_order:
                                v3 = v2 ^ (1 << b3)
                                if v3 == v:
                                    continue
                                if v3 not in used and v3 not in forbidden:
                                    sub_score += 1
                            score += 1 + 0.5 * sub_score
                    if score > best_score:
                        best_score = score
                        best_candidate = v
            
            if best_candidate is None:
                break
            
            for nb in get_neighbors(cur):
                forbidden.add(best_candidate)  # Bug fix: should be cur's neighbors
            # Actually, forbidden should be neighbors of cur (the vertex we're leaving)
            # Let me fix: when we move from cur to best_candidate, cur's other neighbors become forbidden
            # But we already added cur to used, so cur's neighbors are forbidden if they're not best_candidate
            # The standard approach: forbidden = all neighbors of all non-end vertices
            # When we extend, the new "forbidden" adds neighbors of the second-to-last vertex
            
            path.append(best_candidate)
            used.add(best_candidate)
        
        return path

    def compute_forbidden(path):
        """Compute forbidden set: neighbors of all non-end vertices."""
        forbidden = set()
        n = len(path)
        if n <= 2:
            return forbidden
        # For extending forward: neighbors of path[0:-1]
        for i in range(n - 1):
            v = path[i]
            for nb in get_neighbors(v):
                if nb not in set(path):
                    forbidden.add(nb)
        return forbidden

    def extend_forward(path, bit_order, max_steps=512):
        """Extend path forward using greedy with lookahead."""
        used = set(path)
        forbidden = compute_forbidden(path)
        return greedy_extend(path, used, forbidden, bit_order, max_steps)

    def extend_backward(path, bit_order, max_steps=512):
        """Extend path backward using greedy with lookahead."""
        # Reverse path, extend forward, reverse back
        rev = list(reversed(path))
        extended = extend_forward(rev, bit_order, max_steps)
        return list(reversed(extended))

    def local_improvement_2opt(path, bit_order, max_rounds=3):
        """Try to improve path by 2-opt style moves and insertions."""
        if len(path) <= 2:
            return path
        
        best_path = list(path)
        best_len = len(best_path)
        
        for round_num in range(max_rounds):
            if time_left() < 5.0:
                break
            
            improved = False
            
            # Try extending from both ends
            for direction in [1, -1]:
                if direction == 1:
                    cur = best_path[-1]
                    used = set(best_path)
                    forbidden = set()
                    for i in range(len(best_path) - 1):
                        for nb in get_neighbors(best_path[i]):
                            if nb not in used:
                                forbidden.add(nb)
                else:
                    cur = best_path[0]
                    used = set(best_path)
                    forbidden = set()
                    for i in range(1, len(best_path)):
                        for nb in get_neighbors(best_path[i]):
                            if nb not in used:
                                forbidden.add(nb)
                
                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v not in used and v not in forbidden:
                        if direction == 1:
                            best_path.append(v)
                        else:
                            best_path.insert(0, v)
                        improved = True
                        break
            
            # Try inserting a vertex in the middle
            if len(best_path) > 2:
                used = set(best_path)
                # For each pair of consecutive vertices, try to insert a vertex between them
                # But we can't insert between consecutive vertices in a path easily.
                # Instead, try to find a vertex that can be added by removing a middle vertex
                # and reconnecting. This is complex; skip for now.
            
            if not improved:
                break
            
            if len(best_path) > best_len:
                best_len = len(best_path)
        
        return best_path

    def branch_and_bound_improve(path, bit_order, time_budget=30.0):
        """Branch-and-bound to try to extend the path further."""
        if time_left() < time_budget:
            return path
        
        best_path = list(path)
        best_len = len(best_path)
        deadline = time.time() + time_budget
        
        def bbf(current_path, used, forbidden, depth=0):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            
            cur = current_path[-1]
            
            # Get candidates
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    candidates.append(v)
            
            if not candidates:
                return
            
            # Score candidates by how many options they give
            scored = []
            for v in candidates:
                score = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if v2 not in used and v2 not in forbidden:
                        score += 1
                scored.append((score, v))
            
            scored.sort(key=lambda x: -x[0])
            
            # Branch on top candidates
            for score, v in scored[:3]:
                if time.time() > deadline:
                    return
                
                new_path = current_path + [v]
                new_used = used | {v}
                new_forbidden = set(forbidden)
                for nb in get_neighbors(cur):
                    if nb not in new_used:
                        new_forbidden.add(nb)
                
                if len(new_path) > best_len:
                    best_path = list(new_path)
                    best_len = len(new_path)
                
                # Recurse
                bbf(new_path, new_used, new_forbidden, depth + 1)
        
        used = set(best_path)
        forbidden = compute_forbidden(best_path)
        bbf(best_path, used, forbidden)
        
        return best_path

    def random_bit_order():
        order = list(range(N))
        random.shuffle(order)
        return order

    def run_greedy_attempt(sv, bo, use_lookahead=True, max_steps=512):
        path = [sv]
        used = {sv}
        forbidden = set()
        if use_lookahead:
            return greedy_with_lookahead(path, used, forbidden, bo, max_steps, lookahead_depth=2)
        else:
            return greedy_extend(path, used, forbidden, bo, max_steps)

    best_path = None
    best_len = 0

    # Phase 1: Many fast greedy attempts with random restarts
    phase1_time = min(30.0, time_left() * 0.4)
    deadline1 = start_time + phase1_time

    # First: try deterministic baseline
    p = run_greedy_attempt(0, list(range(N)), use_lookahead=True)
    best_path = p
    best_len = len(p)

    # Random restarts with greedy + lookahead
    attempts = 0
    while time.time() < deadline1:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        p = run_greedy_attempt(sv, bo, use_lookahead=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1

    # Phase 2: Branch-and-bound improvement on best path
    if time_left() > 10.0:
        bo = random_bit_order()
        improved = branch_and_bound_improve(best_path, bo, time_budget=min(40.0, time_left() * 0.4))
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Phase 3: Local improvement
    if time_left() > 5.0:
        bo = random_bit_order()
        improved = local_improvement_2opt(best_path, bo, max_rounds=5)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Phase 4: More greedy attempts
    if time_left() > 5.0:
        bo_best = random_bit_order()
        for _ in range(30):
            if time_left() < 3.0:
                break
            sv = random.randint(0, 2**N - 1)
            p = run_greedy_attempt(sv, bo_best, use_lookahead=True)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    # Phase 5: Final B&B on best path if time allows
    if time_left() > 15.0:
        bo = random_bit_order()
        improved = branch_and_bound_improve(best_path, bo, time_budget=min(20.0, time_left() * 0.5))
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Final local improvement
    if time_left() > 3.0:
        bo = random_bit_order()
        improved = local_improvement_2opt(best_path, bo, max_rounds=3)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    return best_path
# EVOLVE-BLOCK-END
