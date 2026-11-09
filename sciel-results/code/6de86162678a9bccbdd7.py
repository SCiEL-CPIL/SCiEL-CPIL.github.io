import random
import time
import sys
from itertools import combinations

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    Returns True if found, False otherwise.
    """
    n = len(weights)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return target == 0
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = set()
    left_sums.add(0)
    for w in left:
        new_sums = set()
        for s in left_sums:
            if s + w <= target:
                new_sums.add(s + w)
        left_sums.update(new_sums)
        
    right_sums = set()
    right_sums.add(0)
    for w in right:
        new_sums = set()
        for s in right_sums:
            if s + w <= target:
                new_sums.add(s + w)
        right_sums.update(new_sums)
        
    # Check intersection
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def is_valid(S):
    """Check if set S is valid (no reciprocal subsum violations)."""
    S_list = list(S)
    for x in S_list:
        target = L // x
        weights = [L // y for y in S_list if y != x]
        if subset_sum_exists(weights, target):
            return False
    return True

def precompute_conflicts():
    """
    Precompute all conflict hyperedges.
    A conflict hyperedge is a set of elements {x, y1, y2, ..., yk} such that
    1/x = sum(1/yi) for some non-empty subset of {y1,...,yk}.
    In the independent set problem, we cannot select all elements of any hyperedge.
    """
    all_elements = list(range(1, 25))
    # For each element x, find all subsets of other elements whose reciprocal sum = 1/x
    # Scale by L: target = L/x, weights = L/y for y != x
    # We need to find all subsets of {L/y : y in all_elements, y != x} that sum to L/x
    
    # Build conflict graph: for each pair (x, S) where S is a subset of elements
    # and 1/x = sum(1/s for s in S), we have a hyperedge {x} ∪ S
    
    # For efficiency, we'll find minimal conflicts.
    # A conflict exists if there's a subset T of other selected elements such that sum(1/t for t in T) = 1/x.
    
    # Let's precompute for each x, which subsets of other elements can form 1/x.
    # We'll store these as frozensets.
    
    conflicts = []  # list of frozensets (hyperedges)
    
    for x in all_elements:
        target = L // x
        others = [y for y in all_elements if y != x]
        weights = [L // y for y in others]
        
        # Find all subsets of weights that sum to target
        # Use meet-in-the-middle to enumerate all such subsets
        n = len(weights)
        mid = n // 2
        left = weights[:mid]
        right = weights[mid:]
        
        # Enumerate all subset sums of left
        left_subsets = {}  # sum -> list of bitmasks
        left_subsets[0] = [0]
        for i, w in enumerate(left):
            new_entries = {}
            for s, masks in list(left_subsets.items()):
                if s + w <= target:
                    if s + w not in new_entries:
                        new_entries[s + w] = []
                    for m in masks:
                        new_entries[s + w].append(m | (1 << i))
            for s, masks in new_entries.items():
                if s in left_subsets:
                    left_subsets[s].extend(masks)
                else:
                    left_subsets[s] = masks
        
        # Enumerate all subset sums of right
        right_subsets = {}  # sum -> list of bitmasks
        right_subsets[0] = [0]
        for i, w in enumerate(right):
            new_entries = {}
            for s, masks in list(right_subsets.items()):
                if s + w <= target:
                    if s + w not in new_entries:
                        new_entries[s + w] = []
                    for m in masks:
                        new_entries[s + w].append(m | (1 << i))
            for s, masks in new_entries.items():
                if s in right_subsets:
                    right_subsets[s].extend(masks)
                else:
                    right_subsets[s] = masks
        
        # Combine to find subsets summing to target
        for ls, lmasks in left_subsets.items():
            rs = target - ls
            if rs in right_subsets:
                for lm in lmasks:
                    for rm in right_subsets[rs]:
                        # Build the subset of other elements
                        subset_mask = (lm << mid) | rm
                        subset_elements = set()
                        for j in range(n):
                            if subset_mask & (1 << j):
                                subset_elements.add(others[j])
                        if subset_elements:
                            hyperedge = frozenset({x} | subset_elements)
                            if hyperedge not in conflicts:
                                conflicts.append(hyperedge)
    
    return conflicts

def branch_and_bound(conflicts, deadline):
    """
    Branch-and-bound to find maximum weight independent set on hypergraph.
    Vertices: 1..24, weight 1 each.
    Hyperedges: cannot select all vertices in any hyperedge.
    """
    n = 24
    # Convert conflicts to a format suitable for B&B
    # For each vertex, store which conflicts it participates in
    vertex_conflicts = [[] for _ in range(n)]
    for i, conf in enumerate(conflicts):
        for v in conf:
            vertex_conflicts[v - 1].append(i)
    
    best_set = set()
    best_size = 0
    
    # Sort vertices by degree (number of conflicts) for better pruning
    # We'll use a simple recursive B&B
    
    def bb(remaining, selected, excluded, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: |selected| + |remaining|
        if len(selected) + len(remaining) <= best_size:
            return
        
        if not remaining:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
            return
        
        # Try to add a vertex from remaining
        # Choose vertex with most conflicts for better pruning
        v = max(remaining, key=lambda x: len(vertex_conflicts[x - 1]))
        
        # Option 1: Include v
        # Check if adding v would violate any conflict
        # A conflict is violated if all its vertices are in selected ∪ {v}
        new_selected = selected | {v}
        valid = True
        for ci in vertex_conflicts[v - 1]:
            conf = conflicts[ci]
            # Check if all vertices in conf are in new_selected
            all_in = True
            for u in conf:
                if u not in new_selected:
                    all_in = False
                    break
            if all_in:
                valid = False
                break
        
        if valid:
            new_remaining = [u for u in remaining if u != v]
            # Prune: if including v forces some vertices to be excluded
            # (because they'd complete a conflict with v and others in selected)
            # For simplicity, just recurse
            bb(new_remaining, new_selected, excluded, depth + 1)
        
        # Option 2: Exclude v
        new_remaining = [u for u in remaining if u != v]
        bb(new_remaining, selected, excluded | {v}, depth + 1)
    
    bb(list(range(1, 25)), set(), set(), 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    all_elements = list(range(1, 25))
    
    # Precompute conflicts
    conflicts = precompute_conflicts()
    
    # Branch and bound
    best_set = branch_and_bound(conflicts, deadline)
    
    # If B&B didn't find anything, fall back to greedy
    if not best_set:
        S = set()
        for x in all_elements:
            if time.time() > deadline:
                break
            test_set = S | {x}
            if is_valid(test_set):
                S = test_set
        best_set = S
    
    # Local search to improve
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(best_set):
                if time.time() > deadline:
                    break
                S_test = best_set - {x}
                for m in [y for y in all_elements if y not in S_test]:
                    if time.time() > deadline:
                        break
                    S_new = S_test | {m}
                    if is_valid(S_new) and len(S_new) > len(best_set):
                        best_set = set(S_new)
                        improved = True
                        break
                if improved:
                    break
    
    if not best_set:
        return [1]
        
    return list(best_set)
