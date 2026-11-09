"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses randomized greedy construction and local search with bitset optimization.
"""

import time
import random
import sys
from functools import lru_cache

# Set recursion limit if needed, though we'll use iterative approaches where possible
sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_independent_set_fast(adj_bits, n, k):
    """
    Fast check for independent set of size k using bitsets and Branch and Bound.
    Uses memoization and a greedy lower bound for pruning.
    """
    memo = {}
    
    def greedy_independent(rem):
        """Greedy: repeatedly pick vertex with min degree in rem."""
        count = 0
        r = rem
        while r:
            # Find vertex with minimum degree in r
            best_v = -1
            best_deg = n + 1
            temp = r
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = _popcount(adj_bits[v] & r)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            count += 1
            r = r & ~(adj_bits[best_v] | (1 << best_v))
        return count
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning: If even taking all remaining vertices doesn't reach k
        if size + _popcount(rem) < k:
            return False
        
        # Pruning: Greedy lower bound on independent set in rem
        # If size + greedy(rem) < k, we can't reach k
        # This is a valid bound because greedy gives a lower bound on alpha(rem)
        # Actually greedy gives an upper bound on the independent set size we can find,
        # but it's not necessarily the true alpha. Let's use a different approach.
        # 
        # Better pruning: If we can find a matching of size (k - size) in the complement...
        # That's complex. Let's stick with the simple bound and the vertex branching.
        
        key = (rem, size)
        if key in memo:
            return memo[key]
        
        # Choose a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the remaining subgraph.
        best_v = -1
        best_deg = -1
        temp_rem = rem
        while temp_rem:
            lsb = temp_rem & -temp_rem
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_rem ^= lsb
            
        if best_v == -1:
            memo[key] = False
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        if search(new_rem, size + 1):
            memo[key] = True
            return True
            
        # Branch 2: Exclude best_v
        new_rem2 = rem & ~v_bit
        if search(new_rem2, size):
            memo[key] = True
            return True
            
        memo[key] = False
        return False
        
    return search((1 << n) - 1, 0)

def _create_adjacency(n):
    """Create adjacency lists and bitsets."""
    adj = [set() for _ in range(n)]
    # Bitsets for fast intersection
    adj_bits = [0] * n
    return adj, adj_bits

def _add_edge(adj, adj_bits, u, v):
    """Add edge u-v, update structures."""
    adj[u].add(v)
    adj[v].add(u)
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _has_c4(adj, adj_bits, n):
    """
    Check if graph contains a C4.
    A C4 exists if there are two vertices u, v (non-adjacent) that have >= 2 common neighbors.
    We check this during construction or verification.
    """
    # This is O(n^2 * (n/word_size)) which is fast for n <= 43
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_bits[u] >> v) & 1:
                continue # Adjacent vertices don't form opposite corners of C4 directly in this check logic
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1): # If common has more than 1 bit set
                return True
    return False

def _count_common_neighbors(adj_bits, u, v):
    """Count common neighbors of u and v."""
    common = adj_bits[u] & adj_bits[v]
    return bin(common).count('1')

def _can_add_edge(adj, adj_bits, u, v, n):
    """
    Check if adding edge (u, v) would create a C4.
    Adding (u, v) creates a C4 if there exists a path u-w-z-v of length 3 that is not using u-v.
    Actually, simpler: u and v already share some neighbors. If they share >= 1 neighbor w, 
    adding edge (u, v) creates a triangle, not a C4 immediately. 
    Wait, the definition of C4-free is: no two vertices have 2 or more common neighbors.
    If we add edge (u, v), it doesn't change the common neighbors of other pairs? 
    No, it changes the adjacency. 
    The condition "No two vertices may have two common neighbours" is the C4-free condition.
    
    If we add edge (u, v):
    1. u and v become adjacent.
    2. For any other pair (x, y), their common neighbors are unchanged UNLESS x=u or y=v etc.
    3. Specifically, we must ensure that after adding (u, v), no pair has >= 2 common neighbors.
    
    Let's check which pairs might violate the condition after adding (u, v):
    - Pair (u, v): They are now adjacent. The condition applies to non-adjacent vertices? 
      The task says: "No two vertices may have two common neighbours (no C4)".
      In graph theory, a C4 is a cycle of length 4. A cycle u-w-z-v-u requires u and v to be non-adjacent? 
      No, u and v are opposite in the cycle u-w-z-v. So u and v are NOT adjacent in the cycle.
      If u and v ARE adjacent, they can't be opposite in a C4. 
      However, if u and v are adjacent, can they have 2 common neighbors? 
      If u and v are adjacent and share neighbors w, z, then u-w-z-v-u is a C4? 
      Yes, if w != z. 
      So, even for adjacent vertices, having 2 common neighbors creates a C4 (u-w-z-v-u).
      Wait, the cycle is u-w-z-v-u. The edges are (u,w), (w,z), (z,v), (v,u).
      So yes, any pair of vertices (adjacent or not) must have at most 1 common neighbor.
      
    Therefore, adding edge (u, v) is safe if and only if:
    1. u and v do not already have 2 common neighbors. (If they have 2, a C4 already exists or will be reinforced? 
       Actually, if they have 2 common neighbors, the graph ALREADY has a C4 (u-w-z-v-u is not a cycle if u-v is not edge, 
       but u-w-z-v is a path. If u-v is not edge, u-w-z-v-u is not a cycle. 
       Wait. C4 is a cycle. 
       If u, v are non-adjacent and have 2 common neighbors w, z, then u-w-z-v-u is a C4? 
       Edges: (u,w), (w,z), (z,v), (v,u)? No, (v,u) is not an edge.
       The cycle would be u-w-z-v... where is the 4th edge? 
       The cycle is u-w-z-v-u. This requires edge (v,u). 
       So if u and v are NON-adjacent, and have 2 common neighbors, do they form a C4? 
       No. The cycle is u-w-z-v. That's a path of length 3. 
       To close the cycle, we need an edge between u and v. 
       So, if u and v are NON-adjacent and have 2 common neighbors, we do NOT necessarily have a C4.
       Example: Star graph K1,3. Center c, leaves l1, l2, l3. 
       l1 and l2 have 1 common neighbor (c). 
       l1 and l3 have 1 common neighbor (c).
       No C4.
       
       Let's re-read the standard definition. 
       A graph is C4-free if it contains no cycle of length 4.
       A cycle of length 4 is a sequence of distinct vertices v1, v2, v3, v4 such that 
       (v1,v2), (v2,v3), (v3,v4), (v4,v1) are edges.
       
       Condition: "No two vertices may have two common neighbours".
       If u and v have common neighbors w and z.
       If u and v are adjacent: Cycle u-w-z-v-u. Edges (u,w), (w,z), (z,v), (v,u). This is a C4.
       If u and v are NOT adjacent: The path u-w-z-v exists. To form a C4, we need a 4th edge. 
       Is there a theorem? 
       "A graph is C4-free if and only if every pair of vertices has at most one common neighbor."
       Let's check.
       If every pair has at most 1 common neighbor:
       Suppose there is a C4: v1-v2-v3-v4-v1.
       v1 and v3 have common neighbors v2 and v4.
       So v1 and v3 have 2 common neighbors.
       This contradicts the condition.
       So YES, the condition "No two vertices may have two common neighbours" is equivalent to C4-free.
       
       Therefore, when adding edge (u, v):
       We must ensure that NO pair of vertices has 2 or more common neighbors.
       
       Which pairs are affected by adding (u, v)?
       Only pairs that include u or v.
       
       1. Pair (u, v):
          Before adding, they had k common neighbors.
          After adding, they are adjacent.
          They still have the same k common neighbors.
          If k >= 2, then (u, v) already violated the condition? 
          No, if they are non-adjacent and have k=2 common neighbors, does that form a C4?
          Let's trace: u-w1-z-v-u? No, u-v is not edge.
          The cycle would be u-w1-v-w2-u?
          If u and v have common neighbors w1 and w2.
          Edges: (u,w1), (w1,v), (v,w2), (w2,u).
          This is a cycle u-w1-v-w2-u.
          Yes! This is a C4.
          So, if u and v (non-adjacent) have 2 common neighbors, a C4 ALREADY EXISTS.
          
          So, if the graph is currently C4-free, then u and v (non-adjacent) have at most 1 common neighbor.
          When we add edge (u, v), they become adjacent.
          Do they still have at most 1 common neighbor? Yes, the common neighbors didn't change.
          So pair (u, v) is safe.
       
       2. Pair (u, x) where x != v:
          Before: Common neighbors of u and x.
          After: v is now a neighbor of u. Is v a neighbor of x?
          If v is a neighbor of x, then v becomes a new common neighbor of u and x.
          So we must check: If v is in adj[x], then u and x gain v as a common neighbor.
          If u and x ALREADY had 1 common neighbor, adding v makes it 2 -> C4 created.
          If u and x ALREADY had 0 common neighbors, adding v makes it 1 -> OK.
          If u and x ALREADY had >= 1 common neighbor, we need to count them.
          
          So, for each neighbor x of v (x != u), we must check if u and x have any common neighbors other than v.
          Actually, simpler: 
          Count common neighbors of u and x BEFORE adding (u,v).
          Let c = common_count(u, x).
          If x is adjacent to v, then after adding (u,v), v becomes a common neighbor of u and x (since v is adj u and v is adj x).
          So new common count = c + 1.
          We need new common count <= 1.
          So we need c <= 0.
          So if x is adjacent to v, u and x must have NO common neighbors.
          
       3. Pair (v, y) where y != u:
          Symmetric to above.
          For each neighbor y of u (y != v), if y is adjacent to v, then u and v... wait.
          Pair (v, y). v is adj to y? 
          We are adding (u, v).
          v's neighbors are unchanged. u's neighbors gain v.
          So common neighbors of v and y change only if y is adjacent to u.
          If y is adjacent to u, then u becomes a new common neighbor of v and y.
          So for each neighbor y of u (y != v), if y is adjacent to v, then v and y gain u as a common neighbor.
          We need common_count(v, y) + 1 <= 1 => common_count(v, y) == 0.
          
       4. Pair (x, y) where x != u, v and y != u, v:
          Unchanged.
       
       So, adding edge (u, v) is safe if and only if:
       - For every x in adj[v] \ {u}: common_count(u, x) == 0
       - For every y in adj[u] \ {v}: common_count(v, y) == 0
       
       Note: common_count(u, x) counts neighbors of u that are also neighbors of x.
       
    """
    # Check neighbors of v
    for x in adj[v]:
        if x == u:
            continue
        # Check if u and x have any common neighbors
        # common neighbors of u and x are (adj_bits[u] & adj_bits[x])
        if adj_bits[u] & adj_bits[x]:
            return False
            
    # Check neighbors of u
    for y in adj[u]:
        if y == v:
            continue
        # Check if v and y have any common neighbors
        # common neighbors of v and y are (adj_bits[v] & adj_bits[y])
        if adj_bits[v] & adj_bits[y]:
            return False
            
    return True

def _independent_set_greedy(adj, n, limit=11):
    """
    Try to find an independent set of size 'limit'.
    Returns the set if found, else None.
    """
    remaining = set(range(n))
    indep = set()
    while remaining:
        best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        indep.add(best_v)
        remaining -= adj[best_v]
        remaining.discard(best_v)
        if len(indep) >= limit:
            return indep
    if len(indep) >= limit:
        return indep
    return None

def _local_search_reduce_alpha(adj, adj_bits, n, target_k, max_iters=500):
    """
    Try to reduce alpha below target_k by adding edges.
    Finds a large independent set, then tries to add edges between its vertices
    that don't create C4.
    Returns True if alpha < target_k after modifications.
    """
    # Find current maximum independent set (approximate)
    # Use greedy to find a large independent set
    remaining = set(range(n))
    indep = []
    while remaining:
        best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        indep.append(best_v)
        remaining -= adj[best_v]
        remaining.discard(best_v)
    
    if len(indep) < target_k:
        return True  # Already good
    
    # Try to add edges between vertices in indep to break the independent set
    # An edge between two vertices in indep removes them from being independent together
    # We want to add edges such that no independent set of size target_k exists
    
    # Strategy: For each pair in indep that can have an edge added safely,
    # add it. This will reduce the size of the maximum independent set.
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        # Re-find independent set
        remaining = set(range(n))
        indep = []
        while remaining:
            best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
            indep.append(best_v)
            remaining -= adj[best_v]
            remaining.discard(best_v)
        
        if len(indep) < target_k:
            return True
        
        # Try to add an edge between two vertices in indep
        for i in range(len(indep)):
            for j in range(i + 1, len(indep)):
                u, v = indep[i], indep[j]
                if v not in adj[u]:  # Not already adjacent
                    if _can_add_edge(adj, adj_bits, u, v, n):
                        _add_edge(adj, adj_bits, u, v)
                        changed = True
                        break
            if changed:
                break
    
    # Check if alpha is now below target_k
    return not _has_independent_set_fast(adj_bits, n, target_k)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_n = 30
    best_edges = []
    
    # Initial fallback: 10 disjoint triangles (n=30)
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    rng = random.Random(42)
    
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
            
        # Dynamic restarts: more for larger n (harder to find valid)
        restarts = 15 if n > 38 else 25
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
                
            adj = [set() for _ in range(n)]
            adj_bits = [0] * n
            
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(pairs)
            
            # Greedy fill: add all safe edges
            for u, v in pairs:
                if time.time() > deadline - 5:
                    break
                if _can_add_edge(adj, adj_bits, u, v, n):
                    _add_edge(adj, adj_bits, u, v)
            
            # Verify C4
            if _has_c4(adj, adj_bits, n):
                continue
            
            # Check if alpha < 11
            if not _has_independent_set_fast(adj_bits, n, 11):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # If alpha >= 11, try local search to reduce alpha
            if _local_search_reduce_alpha(adj, adj_bits, n, 11, max_iters=200):
                if not _has_independent_set_fast(adj_bits, n, 11):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Constructed graph with n={res['n']}")
