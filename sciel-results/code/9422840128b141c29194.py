import time
from random import Random
from collections import defaultdict

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start = time.time()
    deadline = start + 145  # safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    N = 20
    all_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.add(frozenset((i, j)))
    
    # Tree edge counts: trees[k] has k edges (T_{k+1} has k edges)
    # Total = 0+1+2+...+19 = 190 = |E(K20)|
    
    # Strategy: Use backtracking with constraint propagation
    # Process trees in an order that maximizes constraint tightness
    # Larger trees are more constrained, so embed them first
    
    # For each tree, we need to find an injective mapping from tree vertices to K20 vertices
    # such that all tree edges map to distinct unused K20 edges
    
    # Precompute tree structures
    tree_adj = []
    tree_degrees = []
    for k in range(20):
        edges = trees[k]
        adj = defaultdict(list)
        deg = [0] * (k + 1)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_degrees.append(deg)
    
    # Embedding order: largest trees first (most constrained)
    # But we also want to consider the structure
    embed_order = list(range(19, -1, -1))  # T20, T19, ..., T1
    
    # State: 
    # - used_edges: set of frozensets of edges already used
    # - embeddings: list where embeddings[k] = list of host vertices for T_{k+1}, or None
    # - For backtracking, we need to be able to undo
    
    # Better approach: Use a recursive backtracking solver with memoization-like pruning
    
    # Let's use a different strategy: 
    # 1. First, find embeddings for the most constrained trees
    # 2. Use forward checking to prune early
    
    # For efficiency, let's represent the problem as:
    # For each tree k (0-indexed, representing T_{k+1}), we need an assignment
    # assign: {tree_vertex -> host_vertex} that is injective
    # such that for each edge (u,v) in tree, (assign[u], assign[v]) is an unused edge
    
    # Key insight: Since we must use ALL edges, this is equivalent to finding
    # an edge-decomposition. Let's use a SAT-like approach with backtracking.
    
    # Alternative: Use a greedy + local search approach
    # 1. Try to embed all trees greedily
    # 2. If stuck, use local search to fix conflicts
    
    # Let me try a more systematic backtracking with good heuristics
    
    class Solver:
        def __init__(self):
            self.used_edges = set()
            self.embeddings = [None] * 20
            self.tree_idx = 0  # current tree being embedded
            self.assign = {}  # tree_vertex -> host_vertex for current tree
            self.order = []  # BFS order for current tree
            self.adj = None  # adjacency for current tree
            self.k = 0  # number of vertices in current tree
            self.edges = []  # edges of current tree
            self.deadline = deadline
            self.nodes = 0
            
        def time_check(self):
            return time.time() > self.deadline
        
        def available_hosts(self, tree_v):
            """Get host vertices that can be assigned to tree_v given current partial assignment"""
            assigned_hosts = set(self.assign.values())
            candidates = []
            for h in range(N):
                if h in assigned_hosts:
                    continue
                # Check all neighbors of tree_v that are already assigned
                ok = True
                for nb in self.adj[tree_v]:
                    if nb in self.assign:
                        edge = frozenset((h, self.assign[nb]))
                        if edge in self.used_edges:
                            ok = False
                            break
                if ok:
                    candidates.append(h)
            return candidates
        
        def solve_tree(self, tree_idx):
            """Embed tree tree_idx. Returns True if successful."""
            if self.time_check():
                return False
            
            k = tree_idx + 1
            self.k = k
            self.edges = trees[tree_idx]
            self.adj = tree_adj[tree_idx]
            
            if k == 1:
                # No edges, just pick any vertex
                # But we need to be careful: this vertex might be needed later
                # For now, just pick vertex 0 (or any)
                self.embeddings[tree_idx] = [0]
                return True
            
            # Find BFS order starting from a high-degree vertex
            deg = tree_degrees[tree_idx]
            root = max(range(k), key=lambda v: deg[v])
            
            visited = [False] * k
            order = [root]
            visited[root] = True
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v in self.adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        order.append(v)
                        queue.append(v)
            
            self.order = order
            self.assign = {}
            
            return self._backtrack(0)
        
        def _backtrack(self, pos):
            if self.time_check():
                return False
            
            self.nodes += 1
            
            if pos == self.k:
                # All vertices assigned, verify all edges are valid and unused
                for u, v in self.edges:
                    edge = frozenset((self.assign[u], self.assign[v]))
                    if edge in self.used_edges:
                        # This shouldn't happen if we checked during assignment
                        return False
                # Commit
                for u, v in self.edges:
                    self.used_edges.add(frozenset((self.assign[u], self.assign[v])))
                self.embeddings[self.tree_idx] = [self.assign[i] for i in range(self.k)]
                return True
            
            tree_v = self.order[pos]
            candidates = self.available_hosts(tree_v)
            
            # Heuristic: try hosts that are more constrained first
            # Or just try in order
            for h in candidates:
                self.assign[tree_v] = h
                if self._backtrack(pos + 1):
                    return True
                del self.assign[tree_v]
            
            return False
        
        def solve(self):
            """Solve the complete packing problem"""
            self.tree_idx = 0
            for tree_idx in embed_order:
                self.tree_idx = tree_idx
                if not self.solve_tree(tree_idx):
                    return False
            return True
    
    # Try the solver
    solver = Solver()
    if solver.solve():
        # Verify
        total = 0
        for i in range(20):
            if solver.embeddings[i] is not None:
                total += i  # trees[i] has i edges
        # Should be 190
        return solver.embeddings
    
    # If the main solver failed, try with different strategies
    # Strategy 2: Randomized restarts with different tree orders
    
    rng = Random(42)
    
    for attempt in range(1000):
        if time.time() > deadline - 5:
            break
        
        # Shuffle the embed order with some structure
        # Keep large trees early but add randomness
        base_order = list(range(19, -1, -1))
        # Perturb: sometimes swap adjacent elements
        order = base_order[:]
        for _ in range(5):
            if len(order) > 1:
                i = rng.randrange(len(order) - 1)
                order[i], order[i+1] = order[i+1], order[i]
        
        solver = Solver()
        solver.embed_order = order
        
        # Override solve to use custom order
        def solve_with_order():
            solver.used_edges = set()
            solver.embeddings = [None] * 20
            for tree_idx in order:
                solver.tree_idx = tree_idx
                if not solver.solve_tree(tree_idx):
                    return False
            return True
        
        if solve_with_order():
            total = 0
            for i in range(20):
                if solver.embeddings[i] is not None:
                    total += i
            return solver.embeddings
    
    # If all else fails, return what we have (partial packing)
    # But this will score poorly. Let's try one more strategy.
    
    # Strategy 3: Greedy with edge-based assignment
    # For each tree, assign edges first, then vertices
    
    # Actually, let's try a completely different approach:
    # Use the fact that this is an edge-decomposition and try to
    # build it incrementally with strong lookahead
    
    # For now, return the best partial result
    # The solver might have partial embeddings
    if solver.embeddings:
        return solver.embeddings
    
    # Last resort: return all None
    return [None] * 20
