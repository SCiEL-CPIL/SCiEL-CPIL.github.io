"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80.0
    
    # Target: Minimize |S| such that lengths 3..N are covered by:
    # 1. N (base)
    # 2. s+1 for s in S
    # 3. N-s+1 for s in S
    # 4. |s_j - s_i| + 2 for s_i, s_j in S
    # 5. N - |s_j - s_i| + 2 for s_i, s_j in S
    
    def covers_all(S):
        covered = set([N])
        for s in S:
            covered.add(s + 1)
            covered.add(N - s + 1)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = abs(S[j] - S[i])
                covered.add(d + 2)
                covered.add(N - d + 2)
        # Check if 3..N-1 are in covered
        for l in range(3, N):
            if l not in covered:
                return False
        return True

    def get_cycle_for_length(L, S):
        # Returns a list of vertices forming a cycle of length L
        if L == N:
            return list(range(N))
        
        # Case 1: s+1
        s = L - 1
        if s in S:
            # Cycle 0->1->...->s->0
            return list(range(s+1))
        
        # Case 2: N-s+1
        s = N - L + 1
        if s in S:
            # Cycle 0->N-1->...->s->0
            return [0] + list(range(N-1, s-1, -1))
        
        # Case 3: d+2
        d = L - 2
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if S[j] - S[i] == d:
                    # Cycle 0->S[i]->...->S[j]->0
                    # Path S[i] to S[j] is S[i], S[i]+1, ..., S[j]
                    return [0] + list(range(S[i], S[j]+1))
        
        # Case 4: N-d+2
        d = N - L + 2
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if S[j] - S[i] == d:
                    # Cycle 0->S[j]->...->S[i]->0 (backward path from S[j] to S[i])
                    # Path S[j] to S[i] backward: S[j], S[j]-1, ..., S[i]
                    # Wait, the base cycle is 0-1-...-N-1-0.
                    # Backward path from S[j] to S[i] is S[j], S[j]-1, ..., S[i].
                    # Edges: (S[j], S[j]-1)...(S[i]+1, S[i]).
                    # Chords: (0, S[j]), (0, S[i]).
                    # Cycle: 0, S[j], S[j]-1, ..., S[i], 0.
                    return [0] + list(range(S[j], S[i]-1, -1))
        
        # If not found, raise error
        raise ValueError(f"Cycle for length {L} not found")

    # Search for minimal S
    best_S = None
    best_size = N # Upper bound
    
    # Greedy initialization
    # Try to find a small set
    # We know m=19 works. Let's try to find smaller.
    
    # Randomized Local Search
    random.seed(42)
    
    # Start with a random set of size 15
    current_S = set(random.sample(range(1, N), 15))
    # Ensure 2 is in S to cover length 3? Not strictly necessary if other diffs cover it, but 2 is good.
    # Actually, length 3 requires d=1 or s=2.
    # If d=1, we need s, s+1 in S.
    # If s=2, we have 2 in S.
    
    # Let's use a greedy approach to build S
    S_greedy = set()
    uncovered = set(range(3, N))
    
    while uncovered:
        best_candidate = None
        best_gain = 0
        for s in range(1, N):
            if s in S_greedy: continue
            new_S = S_greedy | {s}
            new_covered = set([N])
            for x in new_S:
                new_covered.add(x+1)
                new_covered.add(N-x+1)
            for i in range(len(new_S)):
                for j in range(i+1, len(new_S)):
                    d = abs(list(new_S)[j] - list(new_S)[i])
                    new_covered.add(d+2)
                    new_covered.add(N-d+2)
            
            gain = len(new_covered & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_candidate = s
        
        if best_candidate is None:
            break
        S_greedy.add(best_candidate)
        # Update uncovered
        new_covered = set([N])
        for x in S_greedy:
            new_covered.add(x+1)
            new_covered.add(N-x+1)
        for i in range(len(S_greedy)):
            for j in range(i+1, len(S_greedy)):
                d = abs(list(S_greedy)[j] - list(S_greedy)[i])
                new_covered.add(d+2)
                new_covered.add(N-d+2)
        uncovered -= new_covered
        
        if time.time() > deadline - 10:
            break

    if covers_all(S_greedy):
        best_S = S_greedy
        best_size = len(S_greedy)
    else:
        # Fallback to known good solution
        best_S = set(range(2, 21))
        best_size = 19

    # Try to improve with local search
    if time.time() < deadline - 5:
        # Try removing elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for s in list(best_S):
                if len(best_S) <= 5: break
                new_S = best_S - {s}
                if covers_all(new_S):
                    best_S = new_S
                    improved = True
                    break
            
            # Try swapping
            for s in list(best_S):
                if time.time() > deadline - 5: break
                for t in range(1, N):
                    if t in best_S: continue
                    new_S = best_S - {s} | {t}
                    if covers_all(new_S):
                        # Check if it's better? Size is same.
                        # Maybe just accept to find a different local min?
                        # Or break to try removal again
                        best_S = new_S
                        improved = True
                        break
                if improved: break

    # Final verification
    if not covers_all(best_S):
        best_S = set(range(2, 21))

    S_list = sorted(list(best_S))
    
    # Construct edges
    cycle_edges = [(i, (i+1) % N) for i in range(N)]
    chords = [(0, s) for s in S_list]
    all_edges = cycle_edges + chords
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Construct cycles
    cycles = []
    for L in range(3, N+1):
        try:
            cyc = get_cycle_for_length(L, S_list)
            # Verify cycle
            if len(cyc) != L:
                raise ValueError
            if len(set(cyc)) != L:
                raise ValueError
            for j in range(L):
                u, v = cyc[j], cyc[(j+1)%L]
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    raise ValueError
            cycles.append(cyc)
        except Exception:
            # Fallback if logic fails
            cycles.append(None)
            break

    if None in cycles:
        # Fallback to the 19-chord solution
        S_list = list(range(2, 21))
        cycle_edges = [(i, (i+1) % N) for i in range(N)]
        chords = [(0, s) for s in S_list]
        all_edges = cycle_edges + chords
        edge_set = set((min(e), max(e)) for e in all_edges)
        
        cycles = []
        for L in range(3, N+1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 20:
                k = L - 1
                cycles.append(list(range(k+1)))
            else:
                k = 41 - L
                cycles.append([0] + list(range(N-1, k-1, -1)))
        
        # Verify
        for i, cyc in enumerate(cycles):
            L = i + 3
            if len(cyc) != L or len(set(cyc)) != L:
                raise ValueError
            for j in range(L):
                u, v = cyc[j], cyc[(j+1)%L]
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    raise ValueError

    return {'edges': [list(e) for e in edge_set], 'cycles': cycles}
# EVOLVE-BLOCK-END
