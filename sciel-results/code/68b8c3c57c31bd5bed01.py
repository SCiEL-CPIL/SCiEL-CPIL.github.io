import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321" # Length 9? 1!+2!+3! = 1+2+6=9. "123121321" is 9.
        # Wait, shortest for n=3 is 9.
    
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Map permutation tuple to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Distance function: cost to append b after a
    # Cost is n - overlap
    def cost(a_idx, b_idx):
        a = perms[a_idx]
        b = perms[b_idx]
        # Max overlap is n-1 (since they are distinct perms, full n overlap implies a==b)
        for k in range(n - 1, 0, -1):
            if a[n-k:] == b[:k]:
                return n - k
        return n

    # 1. Greedy Construction (Nearest Neighbor)
    visited = [False] * N
    path = []
    
    # Start with a random permutation
    current = random.randint(0, N - 1)
    visited[current] = True
    path.append(current)
    
    # Heuristic: To make it deterministic and robust, we can try a few random starts
    # or use a specific start. Let's use a simple start: (1, 2, ..., n)
    # Actually, random is better to escape local structures if we do local search later.
    # But for reproducibility, let's fix the seed or use the identity permutation.
    # The classic construction starts with 1, then inserts 2, etc.
    # Let's just use the identity permutation as start for determinism, 
    # but random restarts in local search will help.
    
    # Let's try a few random starts and keep the best NN path
    best_path = None
    best_cost = float('inf')
    
    max_starts = 10
    for _ in range(max_starts):
        if time.time() > deadline - 60: break # Reserve time for local search
        
        visited = [False] * N
        current = random.randint(0, N - 1)
        visited[current] = True
        path = [current]
        current_cost = 0
        
        for _ in range(N - 1):
            next_node = -1
            min_cost = float('inf')
            
            # Find nearest unvisited
            for j in range(N):
                if not visited[j]:
                    c = cost(current, j)
                    if c < min_cost:
                        min_cost = c
                        next_node = j
            
            if next_node == -1:
                break # Should not happen
            
            visited[next_node] = True
            path.append(next_node)
            current_cost += min_cost
            current = next_node
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_path = path[:]
            
    if best_path is None:
        # Fallback to classic if greedy fails or times out
        return _classic(n)

    path = best_path
    
    # 2. Local Search: 2-Opt
    # Try to improve the path by reversing segments
    # Cost of path = sum(cost(path[i], path[i+1]))
    
    def get_path_cost(p):
        total = 0
        for i in range(len(p) - 1):
            total += cost(p[i], p[i+1])
        return total
    
    current_path_cost = get_path_cost(path)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # 2-Opt: reverse a segment
        # To be efficient, we can limit the range of i, j or use random sampling
        # For N=5040, full 2-opt is O(N^2) per pass, which is too slow.
        # We will use a random 2-opt or limited 2-opt.
        
        # Let's try random 2-opt moves
        for _ in range(1000): # Limit iterations per pass to avoid infinite loop
            if time.time() > deadline - 10: break
            
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Calculate delta cost
            # Removing edges (i-1, i) and (j, j+1)
            # Adding edges (i-1, j) and (i, j+1)
            
            # Handle boundaries
            if i == 0:
                # Remove (j, j+1)
                # Add (0, j) ? No, path is 0..j reversed?
                # 2-Opt on a path (not cycle):
                # Path: ... A, B ... C, D ...
                # Reverse B..C
                # New: ... A, C ... B, D ...
                # Edges removed: (A,B) and (C,D)
                # Edges added: (A,C) and (B,D)
                
                # In our path:
                # Segment [i, j] is reversed.
                # Node before i is i-1 (if i>0)
                # Node after j is j+1 (if j<N-1)
                
                # If i == 0:
                # Removed: (path[j], path[j+1]) if j < N-1
                # Added: (path[0], path[j]) ? No.
                # Original: P0, P1, ..., Pi, ..., Pj, Pj+1
                # Reversed Pi..Pj: P0, ..., Pj, ..., Pi, ..., Pj+1
                # Edges changed:
                # Old: (Pi-1, Pi) and (Pj, Pj+1)
                # New: (Pi-1, Pj) and (Pi, Pj+1)
                
                # Case i=0:
                # Old: (P0, P1) is NOT removed? 
                # The segment is P0..Pj.
                # Reversed: Pj..P0.
                # New path: Pj, ..., P0, Pj+1
                # Edge removed: (Pj, Pj+1)
                # Edge added: (P0, Pj+1)
                # Wait, the internal edges of the segment are reversed, but their cost is symmetric?
                # cost(A,B) != cost(B,A) generally?
                # cost(A,B) = n - overlap(suffix A, prefix B)
                # cost(B,A) = n - overlap(suffix B, prefix A)
                # They are NOT symmetric.
                # So reversing a segment changes the cost of all internal edges.
                # This makes 2-Opt expensive to evaluate exactly.
                
                # Alternative: Use "Or-opt" or "Insertion" moves which are easier to evaluate.
                # Or, just accept that 2-Opt is hard and use a simpler heuristic:
                # Try to swap adjacent nodes?
                # Or, use a "Greedy Repair" or "Simulated Annealing" on the string directly?
                
                pass # Skip complex 2-opt for now, too slow/complex to implement correctly with asymmetric costs in short time.
                
    # Fallback: Just return the Greedy path string
    # Convert path to string
    word = []
    for i in range(len(path)):
        p = perms[path[i]]
        if i == 0:
            word.extend(p)
        else:
            prev_p = perms[path[i-1]]
            # Find overlap
            max_overlap = 0
            for k in range(n - 1, 0, -1):
                if prev_p[n-k:] == p[:k]:
                    max_overlap = k
                    break
            word.extend(p[max_overlap:])
            
    return "".join(map(str, word))

def _classic(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
