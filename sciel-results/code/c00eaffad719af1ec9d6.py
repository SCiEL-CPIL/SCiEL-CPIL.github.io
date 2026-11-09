import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Analytical approach: find minimum chord set for fan graph
    # A chord (0,k) covers cycle lengths k+1 and 41-k
    # A pair of chords (0,k1), (0,k2) with k1<k2 covers cycle length (k2-k1+2)
    # via cycle 0-k1-(k1+1)-...-k2-0
    
    # We need to cover all lengths 3..39 (length 40 is the ring)
    # Let's find a small set of chords that covers all lengths
    
    # Strategy: try all subsets of a candidate set, find minimum that covers all lengths
    # Candidate chords: 2..20 (19 chords)
    # But we can also consider chords beyond 20: chord (0,k) for k>20 covers k+1 and 41-k
    # For k=21: covers 22 and 20
    # For k=22: covers 23 and 19
    # etc.
    
    # Let's try to find a minimum set via greedy + local search
    
    def chord_covers(k):
        """Return set of cycle lengths covered by chord (0,k) alone"""
        l1 = k + 1
        l2 = 41 - k
        s = set()
        if 3 <= l1 <= 39:
            s.add(l1)
        if 3 <= l2 <= 39:
            s.add(l2)
        return s
    
    def pair_covers(k1, k2):
        """Return set of cycle lengths covered by pair of chords (0,k1),(0,k2) via 0-k1-...-k2-0"""
        if k1 > k2:
            k1, k2 = k2, k1
        L = k2 - k1 + 2
        if 3 <= L <= 39:
            return {L}
        return set()
    
    def all_covered(chord_set):
        """Check if chord_set covers all lengths 3..39"""
        covered = set()
        for k in chord_set:
            covered |= chord_covers(k)
        for k1 in chord_set:
            for k2 in chord_set:
                if k1 != k2:
                    covered |= pair_covers(k1, k2)
        return covered == set(range(3, 40))
    
    # Try to find a small chord set
    # Start with all 19 chords, try removing
    full_set = set(range(2, 21))
    
    # Greedy removal
    current = set(full_set)
    for k in sorted(list(current), key=lambda x: random.random()):
        if time.time() > deadline - 10.0:
            break
        if k in current:
            current.discard(k)
            if not all_covered(current):
                current.add(k)
    
    # Local search: try swapping
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for k in list(current):
            for k2 in range(2, 21):
                if k2 in current or k2 == k:
                    continue
                trial = set(current)
                trial.discard(k)
                trial.add(k2)
                if all_covered(trial) and len(trial) < len(current):
                    current = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Build edges
    edges = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))
    for k in current:
        edges.add((0, k))
    
    # Build cycle certificates
    cycles = {}
    cycles[40] = list(range(N))
    
    for k in current:
        l1 = k + 1
        l2 = 41 - k
        if 3 <= l1 <= 39 and l1 not in cycles:
            cycles[l1] = list(range(k + 1))
        if 3 <= l2 <= 39 and l2 not in cycles:
            cycles[l2] = [0] + list(range(N - 1, k - 1, -1))
    
    for k1 in current:
        for k2 in current:
            if k1 >= k2:
                continue
            L = k2 - k1 + 2
            if 3 <= L <= 39 and L not in cycles:
                # Cycle: 0, k1, k1+1, ..., k2, 0
                cycle = [0] + list(range(k1, k2 + 1))
                cycles[L] = cycle
    
    # Check all lengths covered
    if len(cycles) < 38:
        # Fall back to full 19-chord set
        current = set(range(2, 21))
        edges = set()
        for i in range(N):
            a, b = i, (i + 1) % N
            edges.add((min(a, b), max(a, b)))
        for k in current:
            edges.add((0, k))
        
        cycles = {}
        cycles[40] = list(range(N))
        for k in current:
            l1 = k + 1
            l2 = 41 - k
            if l1 not in cycles:
                cycles[l1] = list(range(k + 1))
            if l2 not in cycles:
                cycles[l2] = [0] + list(range(N - 1, k - 1, -1))
        for k1 in current:
            for k2 in current:
                if k1 >= k2:
                    continue
                L = k2 - k1 + 2
                if L not in cycles:
                    cycle = [0] + list(range(k1, k2 + 1))
                    cycles[L] = cycle
    
    final_cycles = []
    for L in range(3, 41):
        final_cycles.append(cycles[L])
    
    edge_list = sorted(edges)
    
    return {'edges': edge_list, 'cycles': final_cycles}

if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}")
    print(f"Cycles: {len(res['cycles'])}")
    edge_set = {frozenset(e) for e in res['edges']}
    for c in res['cycles']:
        assert len(c) == len(set(c)), f"Duplicate vertex in cycle of length {len(c)}"
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            assert frozenset((u, v)) in edge_set, f"Missing edge ({u},{v}) in cycle of length {len(c)}"
    print("All cycles valid.")
