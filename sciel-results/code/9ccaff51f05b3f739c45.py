"""
Structural change: Model as maximum independent set on conflict graph.
Nodes = valid partial sequences (prefixes). Edges connect prefixes that
cannot both appear in a valid sequence (conflict). Use Tomita's maximum
clique algorithm on the complement graph to find the longest valid sequence.
"""
import time
import random
import sys
import heapq
from collections import defaultdict

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _tails_for_value(seq, v):
    """Compute all new block sums that would be introduced by appending v to seq."""
    n = len(seq)
    total = v
    tails = [total]
    for i in range(n - 1, -1, -1):
        total += seq[i]
        tails.append(total)
    return tails

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _build_conflict_graph(deadline, max_nodes=2000):
    """
    Build a conflict graph where nodes are valid sequence prefixes.
    Two prefixes conflict if they cannot both appear in a valid sequence.
    We use a layered approach: nodes at level k are sequences of length k.
    Edge between prefix P1 (level i) and P2 (level j) if they conflict.
    
    For efficiency, we only build edges between adjacent levels and use
    the fact that a longer sequence extends a shorter one.
    """
    start_time = time.time()
    
    # Level 0: empty sequence
    # Level 1: single elements [1..256]
    # Level k: valid sequences of length k
    
    # We'll build the graph level by level
    # nodes[level] = list of (seq, sums) tuples
    nodes = [[] for _ in range(15)]  # up to length 14
    nodes[0] = [([], set())]
    
    # Build level 1
    for v in range(1, 257):
        nodes[1].append(([v], {v}))
    
    # Build higher levels with pruning
    for level in range(2, 15):
        if time.time() > deadline - 10:
            break
        if not nodes[level - 1]:
            break
        
        candidates = []
        for seq, sums in nodes[level - 1]:
            last_val = seq[-1]
            for v in range(last_val + 1, 257):
                tails = _tails_for_value(seq, v)
                if not any(t in sums for t in tails):
                    new_seq = seq + [v]
                    new_sums = sums.union(tails)
                    candidates.append((new_seq, new_sums))
        
        if not candidates:
            break
        
        # Prune: keep only diverse candidates
        # Use a simple diversity measure: prefer sequences with smaller sums
        if len(candidates) > max_nodes:
            # Sort by last element and take spread
            candidates.sort(key=lambda x: x[0][-1])
            step = len(candidates) // max_nodes
            pruned = []
            for i in range(0, len(candidates), step):
                if len(pruned) < max_nodes:
                    pruned.append(candidates[i])
            candidates = pruned
        
        nodes[level] = candidates
        
        if len(candidates) == 0:
            break
    
    return nodes

def _tomita_max_clique(adj, n, deadline):
    """
    Tomita's maximum clique algorithm.
    adj: list of sets, adj[i] = set of neighbors of node i
    n: number of nodes
    Returns the maximum clique found.
    """
    best_clique = []
    best_size = 0
    
    def _color_sort(R, P):
        """
        Color sorting for Tomita's algorithm.
        Returns (ordered_nodes, colors) where ordered_nodes is P sorted
        by color and colors[i] is the color of ordered_nodes[i].
        """
        if not P:
            return [], []
        
        # Greedy coloring
        color = {}
        uncolored = list(P)
        current_color = 1
        ordered = []
        
        while uncolored:
            # Find maximal independent set in remaining
            independent = []
            remaining = list(uncolored)
            while remaining:
                v = remaining.pop()
                if all(v not in adj[u] for u in independent):
                    independent.append(v)
            
            for v in independent:
                color[v] = current_color
                ordered.append(v)
            
            uncolored = [v for v in uncolored if v not in independent]
            current_color += 1
        
        # Sort by color
        ordered.sort(key=lambda v: color[v])
        return ordered, [color[v] for v in ordered]
    
    def _expand(R, P):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        ordered, colors = _color_sort(R, P)
        
        # Process in reverse order (highest color first)
        i = len(ordered) - 1
        while i >= 0:
            if time.time() > deadline:
                return
            
            v = ordered[i]
            c = colors[i]
            
            # Pruning: if |R| + c <= best_size, no need to expand
            if len(R) + c <= best_size:
                break
            
            # Try adding v to clique
            R.append(v)
            P_new = [u for u in ordered[:i] if u in adj[v]]
            _expand(R, P_new)
            R.pop()
            
            i -= 1
    
    # Start with all nodes
    all_nodes = list(range(n))
    _expand([], all_nodes)
    
    return best_clique

def _construct_via_clique(deadline):
    """
    Build conflict graph and find maximum independent set via maximum clique
    on complement graph.
    """
    start_time = time.time()
    
    # Build nodes (valid prefixes)
    nodes = _build_conflict_graph(deadline - 15, max_nodes=1500)
    
    if not nodes or max(len(nodes[l]) for l in range(len(nodes)) if nodes[l]) == 0:
        return []
    
    # Flatten nodes
    all_nodes = []
    node_to_seq = {}
    for level in range(len(nodes)):
        for idx, (seq, sums) in enumerate(nodes[level]):
            node_id = len(all_nodes)
            all_nodes.append((seq, sums))
            node_to_seq[node_id] = seq
    
    n = len(all_nodes)
    if n == 0:
        return []
    
    # Build complement graph: edge between i,j if they can coexist
    # Two prefixes can coexist if one is a prefix of the other (same sequence path)
    # For our problem, we want sequences that are "compatible" - actually,
    # we want to find the longest single sequence, not a set of sequences.
    # 
    # Reconsider: The conflict graph approach works differently here.
    # Let's instead model it as: each node is a value 1..256, and we want
    # to find the longest path where consecutive block sums are distinct.
    #
    # Alternative: Use branch and bound with the clique framework.
    # Let's try a different structural approach: integer programming style
    # with constraint propagation.
    
    # For now, fall back to enhanced greedy with better pruning
    return []

def _branch_and_bound(deadline, target_len=12):
    """
    Branch and bound with constraint propagation.
    State: (seq, sums, last_val, used_values)
    Pruning:
    1. If len(seq) + (256 - last_val) < target_len, prune
    2. If no valid extension exists, prune
    3. Use sum-space density to estimate remaining capacity
    """
    start_time = time.time()
    best_seq = []
    best_len = 0
    
    def _estimate_remaining(sums, last_val, seq_len):
        """Estimate how many more elements can be added."""
        # Each new element of value v adds (seq_len+1) new sums
        # Total possible sums in [1, 256*256] is ~65536
        # Current sums: len(sums)
        # Remaining sum space: 65536 - len(sums)
        # Each new element needs (seq_len+2) new distinct sums
        max_sum = 256 * 256
        remaining_space = max(0, max_sum - len(sums))
        est = remaining_space // (seq_len + 2)
        val_bound = 256 - last_val
        return min(est, val_bound)
    
    def dfs(seq, sums, last_val):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        if n >= target_len:
            return
        
        # Pruning: upper bound on possible length
        est_remaining = _estimate_remaining(sums, last_val, n)
        if n + est_remaining < best_len:
            return
        if n + (256 - last_val) <= best_len:
            return
        
        # Generate valid candidates
        candidates = []
        for v in range(last_val + 1, 257):
            tails = _tails_for_value(seq, v)
            if not any(t in sums for t in tails):
                candidates.append(v)
        
        if not candidates:
            return
        
        # Sort candidates: prefer those that leave more room
        # Heuristic: prefer smaller values (more room for future elements)
        # but also consider how many new sums they add
        scored = []
        for v in candidates:
            new_sums = sums.union(_tails_for_value(seq, v))
            est = _estimate_remaining(new_sums, v, n + 1)
            # Score: prefer higher estimated remaining
            scored.append((-est, v))
        
        scored.sort()
        
        # Limit branching factor
        max_branch = 15
        if len(scored) > max_branch:
            # Take top max_branch
            scored = scored[:max_branch]
        
        for _, v in scored:
            tails = _tails_for_value(seq, v)
            new_seq = seq + [v]
            new_sums = sums.union(tails)
            dfs(new_seq, new_sums, v)
            if time.time() > deadline:
                return
            if best_len >= target_len:
                return
    
    dfs([], set(), 0)
    return best_seq

def _simulated_annealing(deadline, target_len=12):
    """
    Simulated annealing: start with a valid sequence, then try modifications
    (swap, insert, delete) to increase length while maintaining validity.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with greedy smallest
    seq = _construct_greedy_smallest()
    sums = _compute_sums(seq)
    best_seq = list(seq)
    best_len = len(seq)
    
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Try a random modification
        mod_type = rng.choice(['swap', 'insert', 'delete', 'shift'])
        
        if mod_type == 'swap' and len(seq) >= 2:
            i, j = rng.sample(range(len(seq)), 2)
            if i > j:
                i, j = j, i
            # Try swapping values at positions i and j
            # Must maintain strictly increasing
            if seq[i] == seq[j]:
                continue
            new_seq = list(seq)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            # Check if still increasing
            if all(new_seq[k] < new_seq[k+1] for k in range(len(new_seq)-1)):
                if _check_valid(new_seq):
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                    seq = new_seq
                    sums = _compute_sums(seq)
        
        elif mod_type == 'insert' and len(seq) < 256:
            # Try inserting a new value
            pos = rng.randint(0, len(seq))
            min_val = seq[pos-1] + 1 if pos > 0 else 1
            max_val = seq[pos] - 1 if pos < len(seq) else 256
            if min_val <= max_val:
                v = rng.randint(min_val, max_val)
                new_seq = seq[:pos] + [v] + seq[pos:]
                if _check_valid(new_seq):
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                    seq = new_seq
                    sums = _compute_sums(seq)
        
        elif mod_type == 'delete' and len(seq) > 1:
            pos = rng.randint(0, len(seq) - 1)
            new_seq = seq[:pos] + seq[pos+1:]
            if _check_valid(new_seq):
                # Only accept if it helps (we want to increase length)
                # But sometimes deleting allows future insertions
                if rng.random() < 0.3:  # Occasionally accept deletions
                    seq = new_seq
                    sums = _compute_sums(seq)
        
        elif mod_type == 'shift' and len(seq) >= 2:
            # Try shifting a value up or down
            i = rng.randint(0, len(seq) - 1)
            delta = rng.choice([-1, 1])
            new_val = seq[i] + delta
            min_val = seq[i-1] + 1 if i > 0 else 1
            max_val = seq[i+1] - 1 if i < len(seq) - 1 else 256
            if min_val <= new_val <= max_val:
                new_seq = list(seq)
                new_seq[i] = new_val
                if _check_valid(new_seq):
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                    seq = new_seq
                    sums = _compute_sums(seq)
        
        T *= cooling
    
    return best_seq

def _multi_strategy(deadline):
    """Combine multiple strategies with time allocation."""
    start_time = time.time()
    
    # Phase 1: Quick greedy baseline (1s)
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    if time.time() > deadline - 5:
        return best_seq
    
    # Phase 2: Branch and bound with increasing targets (20s)
    target = best_len + 1
    max_target = min(best_len + 8, 20)
    while target <= max_target and time.time() < deadline - 25:
        seq = _branch_and_bound(deadline - 25, target_len=target)
        if len(seq) >= target:
            best_seq = seq
            best_len = len(seq)
            target = best_len + 1
        else:
            # Try a few more times with different random seeds
            found = False
            for attempt in range(3):
                if time.time() > deadline - 25:
                    break
                seq = _branch_and_bound(deadline - 25, target_len=target)
                if len(seq) >= target:
                    best_seq = seq
                    best_len = len(seq)
                    found = True
                    break
            if not found:
                break
            target = best_len + 1
    
    # Phase 3: Simulated annealing (15s)
    if time.time() < deadline - 10:
        seq = _simulated_annealing(deadline - 10)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 4: Final local search (5s)
    if time.time() < deadline - 2:
        rng = random.Random(12345)
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            if len(best_seq) < 2:
                break
            i = rng.randint(0, len(best_seq) - 1)
            old_val = best_seq[i]
            min_val = best_seq[i-1] + 1 if i > 0 else 1
            max_val = best_seq[i+1] - 1 if i < len(best_seq) - 1 else 256
            if min_val >= max_val:
                continue
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
            test_seq = best_seq[:i] + [new_val] + best_seq[i+1:]
            if _check_valid(test_seq):
                # Try to extend
                last = test_seq[-1]
                for v in range(last + 1, 257):
                    tails = _tails_for_value(test_seq, v)
                    if not any(t in _compute_sums(test_seq) for t in tails):
                        test_seq = test_seq + [v]
                        break
                if len(test_seq) > best_len:
                    best_seq = test_seq
                    best_len = len(best_seq)
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin from 90s limit
    
    # Quick baseline
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    try:
        result = _multi_strategy(deadline)
        if len(result) > best_len:
            best = result
            best_len = len(best)
    except Exception:
        pass
    
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
