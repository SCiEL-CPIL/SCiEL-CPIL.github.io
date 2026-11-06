import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    def is_pancyclic(S):
        # S is a set of integers
        lengths = {N} # The rim
        # Add chord lengths
        for k in S:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        # Add pair lengths
        for u, v in itertools.combinations(S, 2):
            lengths.add(abs(u - v) + 2)
        
        # Check if 3..N are covered
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def greedy_find_S(max_steps=100):
        candidates = list(range(2, N - 1)) # 2 to 38
        uncovered = set(range(3, N)) # 3 to 39
        S = set()
        
        # Precompute self lengths for speed
        self_lengths = {k: {k + 1, N - k + 1} for k in candidates}
        
        while uncovered and len(S) < 20:
            best_k = None
            best_gain = -1
            best_new = set()
            
            # Randomize order to escape local minima? Or just iterate.
            # Let's iterate all, pick max gain.
            for k in candidates:
                if k in S:
                    continue
                
                # Calculate new lengths covered by adding k
                new_lens = set(self_lengths[k])
                for v in S:
                    new_lens.add(abs(k - v) + 2)
                
                # Filter to uncovered
                gain = new_lens & uncovered
                if len(gain) > best_gain:
                    best_gain = len(gain)
                    best_k = k
                    best_new = gain
            
            if best_k is None or best_gain == 0:
                break
            
            S.add(best_k)
            uncovered -= best_new
            
            if is_pancyclic(S):
                return S
                
        if is_pancyclic(S):
            return S
        return None

    def find_S(m):
        # Search for a set S of size m using greedy with random restarts
        candidates = list(range(2, N - 1))
        
        attempts = 0
        max_attempts = 2000 if m <= 10 else 5000
        
        best_S = None
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            # Greedy with random tie-breaking
            S = set()
            uncovered = set(range(3, N))
            self_lengths = {k: {k + 1, N - k + 1} for k in candidates}
            
            while uncovered and len(S) < m + 2:
                # Calculate gains for all candidates
                gains = []
                for k in candidates:
                    if k in S:
                        continue
                    new_lens = set(self_lengths[k])
                    for v in S:
                        new_lens.add(abs(k - v) + 2)
                    gain = len(new_lens & uncovered)
                    if gain > 0:
                        gains.append((gain, k))
                
                if not gains:
                    break
                
                # Pick top 3 random or just max?
                # To diversify, pick from top 5 random
                gains.sort(reverse=True)
                top_k = gains[0][1]
                if len(gains) > 1 and random.random() < 0.3:
                    # Pick randomly from top 5
                    top_count = min(5, len(gains))
                    top_k = random.choice([g[1] for g in gains[:top_count]])
                
                S.add(top_k)
                
                # Update uncovered
                new_lens = set(self_lengths[top_k])
                for v in S:
                    if v != top_k:
                        new_lens.add(abs(top_k - v) + 2)
                uncovered -= new_lens
                
                if len(S) >= m and is_pancyclic(S):
                    if best_S is None or len(S) < len(best_S):
                        best_S = set(S)
                    if len(S) == m:
                        return S
                if not uncovered:
                    if best_S is None or len(S) < len(best_S):
                        best_S = set(S)
                    if len(S) <= m:
                        return S
            
            if is_pancyclic(S) and (best_S is None or len(S) < len(best_S)):
                best_S = set(S)
                
        return best_S

    def build_result(S):
        edges = []
        # Rim
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chords
        for k in S:
            edges.append((0, k))
        
        # Generate cycles
        cycles = []
        # Pre-compute which length is covered by what
        # Map length -> (type, params)
        length_map = {}
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)
                
        for u, v in itertools.combinations(S, 2):
            l = abs(u - v) + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', min(u, v), max(u, v))
                
        # Rim
        length_map[N] = ('rim',)
        
        # Construct cycles 3..N
        for L in range(3, N + 1):
            if L not in length_map:
                # Should not happen if S is valid
                return None
            
            typ = length_map[L]
            if typ[0] == 'rim':
                cycles.append(list(range(N)))
            elif typ[0] == 'chord_short':
                k = typ[1]
                # 0, 1, ..., k
                cycles.append([0] + list(range(1, k + 1)))
            elif typ[0] == 'chord_long':
                k = typ[1]
                # 0, N-1, ..., k
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif typ[0] == 'chord_pair':
                u, v = typ[1], typ[2]
                # 0, u, u+1, ..., v
                cycles.append([0, u] + list(range(u + 1, v + 1)))
                
        return {'edges': edges, 'cycles': cycles}

    # Search for minimal m
    # Try smaller m first
    for m in [6, 7, 8, 9, 10, 11]:
        if time.time() > deadline:
            break
        S = find_S(m)
        if S:
            result = build_result(S)
            if result:
                return result

    # Fallback: Fan Graph
    # Hub 0, Path 1..39
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        # Cycle 0, 1, ..., L-1, 0?
        # Fan graph cycles: 0 - i - ... - j - 0. Length j-i+2.
        # We need length L.
        # j - i + 2 = L => j = i + L - 2.
        # Min i=1. Max j=39.
        # i=1 => j = L-1.
        # Cycle: 0, 1, ..., L-1, 0.
        # Vertices: 0, 1, ..., L-1.
        # Edges: (0,1), (1,2)...(L-2, L-1), (L-1, 0).
        # (L-1, 0) is a spoke. Exists.
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}
