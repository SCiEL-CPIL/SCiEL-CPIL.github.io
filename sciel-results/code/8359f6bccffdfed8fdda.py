import random
import math
import time

# Precompute squares up to 256+256 = 512
SQUARES = set()
for i in range(1, 23):  # 22^2 = 484, 23^2 = 529
    SQUARES.add(i * i)

# Precompute conflict graph
# conflict[u] is a set of v such that u+v is a square
# We only need to check pairs (u, v) with u <= v
# For local search, it's easier to have a boolean matrix or set of edges
# 256 nodes is small enough for a 256x256 boolean matrix or list of sets
conflicts = [set() for _ in range(257)]
for u in range(1, 257):
    for v in range(u, 257):
        if u + v in SQUARES:
            conflicts[u].add(v)
            if u != v:
                conflicts[v].add(u)

def is_valid(S):
    S_set = set(S)
    for a in S:
        for b in S:
            if a <= b:
                if a + b in SQUARES:
                    return False
    return True

def count_violations(S):
    S_set = set(S)
    viol = 0
    for a in S:
        for b in S:
            if a <= b:
                if a + b in SQUARES:
                    viol += 1
    return viol

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe
    
    # Initial set: Greedy
    # Try to build a large independent set
    # Randomize the order to get different starting points
    nodes = list(range(1, 257))
    
    best_set = []
    best_size = 0
    
    # Run multiple random restarts
    while time.time() < deadline:
        random.shuffle(nodes)
        S = set()
        for u in nodes:
            # Check if u can be added
            can_add = True
            for v in S:
                if u + v in SQUARES:
                    can_add = False
                    break
            if can_add:
                S.add(u)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Local search from this starting point
        S_set = set(S)
        current_size = len(S_set)
        
        # Local search loop
        while time.time() < deadline:
            improved = False
            # Try adding a node
            candidates = [u for u in range(1, 257) if u not in S_set]
            if candidates:
                random.shuffle(candidates)
                for u in candidates:
                    # Check if u conflicts with any in S_set
                    conflict = False
                    for v in S_set:
                        if u + v in SQUARES:
                            conflict = True
                            break
                    if not conflict:
                        S_set.add(u)
                        current_size += 1
                        improved = True
                        break
            
            # Try removing a node to allow adding others? 
            # Actually, just adding is a hill climb. 
            # To escape local optima, we might need to remove.
            # Let's try a simple swap or removal if no add is possible.
            
            if not improved:
                # Try removing a random node and see if we can add more
                if S_set:
                    to_remove = random.choice(list(S_set))
                    S_set.remove(to_remove)
                    current_size -= 1
                    # Now try to add
                    candidates = [u for u in range(1, 257) if u not in S_set]
                    random.shuffle(candidates)
                    added_any = False
                    for u in candidates:
                        conflict = False
                        for v in S_set:
                            if u + v in SQUARES:
                                conflict = True
                                break
                        if not conflict:
                            S_set.add(u)
                            current_size += 1
                            added_any = True
                            break
                    if not added_any or current_size <= best_size:
                         # If we didn't improve or got worse, we might revert? 
                         # For now, let's just accept the state and continue.
                         pass
            
            if current_size > best_size:
                best_size = current_size
                best_set = list(S_set)
                
            # If we've hit a local optimum where no single add works, 
            # and we've done a few removals, we can break and restart.
            # This is a very naive local search. 
            # Let's add a limit to local search steps to allow restarts.
            # Actually, the loop condition `while time.time() < deadline` handles it,
            # but it might get stuck. 
            # Let's force a restart after 100 steps of local search if no improvement.
            # For simplicity and robustness, let's just do greedy + random restarts.
            # The greedy is fast. 
            # Let's break out of local search after a while to try a new restart.
            # A simple heuristic: if we haven't improved in 50 steps, break.
            # But tracking steps adds overhead. 
            # Given 50s, we can do many restarts.
            break 

    if not best_set:
        # Fallback
        return list(range(1, 257, 3))
        
    return best_set

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe
    
    # Precompute conflicts for speed
    # conflicts[u] is a set of v such that u+v is square
    # We already have this global, but let's ensure it's fast.
    
    best_set = []
    best_size = 0
    
    # Strategy: Randomized Greedy + Local Search
    # 1. Generate a random permutation of 1..256
    # 2. Greedily add nodes if no conflict
    # 3. Local Search: Try to add any node. If none, try to remove one and add two? 
    #    Simpler: Just try to add. If stuck, restart.
    #    To improve, we can do "shake": remove k random nodes, then greedily add back.
    
    while time.time() < deadline:
        nodes = list(range(1, 257))
        random.shuffle(nodes)
        
        S = set()
        for u in nodes:
            # Check conflicts with current S
            ok = True
            for v in S:
                if u + v in SQUARES:
                    ok = False
                    break
            if ok:
                S.add(u)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any node not in S
            for u in range(1, 257):
                if u in S:
                    continue
                ok = True
                for v in S:
                    if u + v in SQUARES:
                        ok = False
                        break
                if ok:
                    S.add(u)
                    improved = True
                    break
            
            if not improved:
                # Try a move: remove one, add two? 
                # Or just remove one random node and try to add again.
                if S:
                    # Pick a node to remove
                    # Prefer nodes with high degree in the induced subgraph? 
                    # Or just random.
                    to_remove = random.choice(list(S))
                    S.remove(to_remove)
                    # Now try to add
                    for u in range(1, 257):
                        if u in S:
                            continue
                        ok = True
                        for v in S:
                            if u + v in SQUARES:
                                ok = False
                                break
                        if ok:
                            S.add(u)
                            improved = True
                            break
                    # If we didn't add anything, we might have lost a node.
                    # Let's check if size decreased.
                    # If size < best_size, we might want to revert, but we don't keep history.
                    # Just continue.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        # If we found a very good set, we can tighten the deadline or just continue.
        # 86 is the seed. We expect to find > 86.
        
    if not best_set:
        return list(range(1, 257, 3))
        
    return best_set
# EVOLVE-BLOCK-END
