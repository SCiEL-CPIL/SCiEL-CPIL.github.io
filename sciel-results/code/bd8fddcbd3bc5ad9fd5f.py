"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE_MARGIN = 5  # seconds before the 90s kill we must stop
_START_TIME = time.time()


def _remaining():
    return 85.0 - (time.time() - _START_TIME)


def _conflict(a, b):
    return (a * b) % (a + b) == 0


def _build_conflict_graph(n=256):
    """Return a list of sets: conflict_graph[i] = set of j != i that conflict with i."""
    nums = list(range(1, n + 1))
    graph = [set() for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            if (a * b) % (a + b) == 0:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _is_valid_set(s, graph, n=256):
    """Check that no two elements in s conflict."""
    sl = sorted(s)
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if j in graph[sl[i] - 1]:  # index is value-1
                return False
    return True


def _greedy_mis(graph, order, n=256):
    """Greedy independent set: iterate in order, pick if not removed."""
    removed = set()
    result = set()
    for v in order:
        if v in removed:
            continue
        result.add(v)
        removed.add(v)
        removed.update(graph[v])
    return result


def _local_search(s, graph, n=256, time_limit=2.0):
    """Try to improve set s via swaps (remove one, add one) and additions."""
    s = set(s)
    best_size = len(s)
    start = time.time()
    
    while time.time() - start < time_limit and _remaining() > _DEADLINE_MARGIN:
        improved = False
        # Try additions
        for v in range(n):
            if v in s:
                continue
            if not (graph[v] & s):
                s.add(v)
                improved = True
        if len(s) > best_size:
            best_size = len(s)
        
        # Try swaps: remove u from s, add v not in s
        if _remaining() < _DEADLINE_MARGIN:
            break
        candidates_to_remove = list(s)
        random.shuffle(candidates_to_remove)
        for u in candidates_to_remove:
            if len(s) <= best_size - 1:
                break
            # Try to find a v that could replace u
            for v in range(n):
                if v in s or v == u:
                    continue
                # Check if v conflicts with s \ {u}
                if not (graph[v] & (s - {u})):
                    # Also make sure removing u and adding v doesn't reduce size
                    # (it keeps same size, but might enable future additions)
                    s.discard(u)
                    s.add(v)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try more aggressive: remove two, add one
            if _remaining() < _DEADLINE_MARGIN:
                break
            done = False
            pairs = list(itertools.combinations(s, 2))
            random.shuffle(pairs)
            for (u1, u2) in pairs[:200]:
                if _remaining() < _DEADLINE_MARGIN:
                    break
                for v in range(n):
                    if v in s or v == u1 or v == u2:
                        continue
                    if not (graph[v] & (s - {u1, u2})):
                        s.discard(u1)
                        s.discard(u2)
                        s.add(v)
                        done = True
                        break
                if done:
                    break
            if not done:
                break
    
    return s


def _seed_set(graph, n=256):
    """Generate a good initial set: all odds, then greedily add evens that don't conflict."""
    s = set(range(1, n + 1, 2))  # all odds
    # Greedily add evens
    evens = list(range(2, n + 1, 2))
    random.shuffle(evens)
    for e in evens:
        if not (graph[e] & s):
            s.add(e)
    return s


def construct():
    t0 = time.time()
    deadline = 85.0  # seconds
    
    # Build conflict graph
    graph = _build_conflict_graph(256)
    
    best = _seed_set(graph, 256)
    best_size = len(best)
    
    # Local search from seed
    if time.time() - t0 < deadline - _DEADLINE_MARGIN:
        improved = _local_search(best, graph, 256, time_limit=min(5.0, deadline - _DEADLINE_MARGIN - (time.time() - t0)))
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Multiple greedy MIS with random orderings
    random.seed(42)
    iterations = 0
    while time.time() - t0 < deadline - _DEADLINE_MARGIN:
        iterations += 1
        # Random order weighted by degree (lower degree first)
        degrees = [len(graph[i]) for i in range(256)]
        # Create weighted random order: prefer low-degree nodes
        weights = [1.0 / (1.0 + d) for d in degrees]
        order = list(range(256))
        # Shuffle with bias
        for i in range(len(order) - 1, 0, -1):
            total_w = sum(weights[j] for j in range(i + 1))
            r = random.random() * total_w
            cum = 0
            for j in range(i + 1):
                cum += weights[j]
                if r <= cum:
                    order[i], order[j] = order[j], order[i]
                    weights[i], weights[j] = weights[j], weights[i]
                    break
        
        result = _greedy_mis(graph, order, 256)
        
        # Local search to improve
        if time.time() - t0 < deadline - _DEADLINE_MARGIN:
            remaining_time = min(3.0, deadline - _DEADLINE_MARGIN - (time.time() - t0))
            if remaining_time > 0.5:
                result = _local_search(result, graph, 256, time_limit=remaining_time)
        
        if len(result) > best_size:
            best = result
            best_size = len(best)
        
        if time.time() - t0 > deadline - _DEADLINE_MARGIN:
            break
    
    # Final local search polish
    if time.time() - t0 < deadline - _DEADLINE_MARGIN:
        remaining_time = max(0.5, deadline - _DEADLINE_MARGIN - (time.time() - t0))
        polished = _local_search(best, graph, 256, time_limit=remaining_time)
        if len(polished) > best_size:
            best = polished
    
    return sorted(best)
# EVOLVE-BLOCK-END
