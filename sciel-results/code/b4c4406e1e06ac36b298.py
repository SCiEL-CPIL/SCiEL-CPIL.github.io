import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18
    
    def build_adj_from_partition(partition):
        """Build adjacency masks from a 3-partite partition where each part is a clique."""
        adj = [0] * n
        # Within each part, all vertices are connected (clique)
        for part in partition:
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, v = part[i], part[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Between parts, no edges (complete 3-partite means edges only within parts)
        # Wait - Turán graph T_3(18) has vertices divided into 3 parts, edges BETWEEN parts
        # Let me reconsider: In T_3(18), each part is independent, edges connect vertices in different parts
        # Actually for K4-free: if we have 3 parts that are cliques, and no edges between them,
        # then any 4 vertices either all in one part (K4!) or spread across parts.
        # Better: use the standard Turán construction - 3 independent sets, complete between them.
        # But that's K_{6,6,6} which is K4-free? No, K_{6,6,6} contains K4.
        # 
        # Correct approach: T_3(18) divides 18 vertices into 3 parts of size 6.
        # Edges exist between vertices in DIFFERENT parts. No edges within a part.
        # This graph is K4-free because any K4 would need 4 mutually adjacent vertices,
        # but vertices in the same part have no edges between them, so at most 3 vertices
        # from 3 different parts can be mutually adjacent... wait that's not right either.
        #
        # Actually in a complete 3-partite graph, any 3 vertices from 3 different parts form a triangle.
        # 4 vertices from 4 different parts would form K4, but we only have 3 parts, so by pigeonhole
        # at least 2 vertices are in the same part and not adjacent. So T_3(18) IS K4-free.
        
        adj = [0] * n
        part_of = [0] * n
        for part_idx, part in enumerate(partition):
            for v in part:
                part_of[v] = part_idx
        
        for u in range(n):
            for v in range(u + 1, n):
                if part_of[u] != part_of[v]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj
    
    def is_tf_mask(mask, adj):
        """Check if vertex set represented by bitmask is triangle-free."""
        m = mask
        while m:
            u_bit = m & (-m)
            u = u_bit.bit_length() - 1
            m ^= u_bit
            common = adj[u] & mask
            while common:
                v_bit = common & (-common)
                v = v_bit.bit_length() - 1
                common ^= v_bit
                if v < u:
                    continue
                w_common = adj[u] & adj[v] & mask
                if w_common:
                    return False
        return True
    
    def max_tf_size_partition(partition):
        """Find max triangle-free induced subgraph size for a 3-partite Turán graph."""
        adj = build_adj_from_partition(partition)
        
        # In a complete 3-partite graph, a triangle-free set can contain:
        # - At most 1 vertex from each pair of parts? No...
        # - A set S is triangle-free iff it doesn't contain a triangle.
        # - In complete 3-partite, a triangle exists iff there's at least one vertex from each of 3 different parts.
        # - So S is triangle-free iff |S ∩ P_i| = 0 for some i, OR S only uses vertices from at most 2 parts.
        # Wait: if S has vertices from all 3 parts, it contains a triangle (pick one from each part).
        # So S is triangle-free iff S is contained in at most 2 parts.
        # Maximum such S = sum of the two largest part sizes.
        
        part_sizes = [len(p) for p in partition]
        part_sizes.sort(reverse=True)
        return part_sizes[0] + part_sizes[1]
    
    def max_tf_size_fast(adj):
        """Fast max TF size check for general graphs."""
        for size in range(12, 4, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_tf_mask(mask, adj):
                    return size
                if time.time() > deadline - 10:
                    return 5
        return 4
    
    def partition_to_edges(partition):
        """Convert a 3-partition to edge list for complete 3-partite graph."""
        edges = set()
        part_of = [0] * n
        for part_idx, part in enumerate(partition):
            for v in part:
                part_of[v] = part_idx
        
        for u in range(n):
            for v in range(u + 1, n):
                if part_of[u] != part_of[v]:
                    edges.add((u, v))
        
        return edges
    
    rng = random.Random(42)
    
    best_score = 18
    best_edges = set()
    
    # Start with balanced partition [6,6,6]
    # Then try local search: move vertices between parts to minimize max(part_i + part_j)
    
    def optimize_partition(initial_partition, max_iters=500):
        """Local search to optimize 3-partition minimizing max pair sum."""
        partition = [list(p) for p in initial_partition]
        current_score = max_tf_size_partition(partition)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            improved = False
            # Try moving a vertex from one part to another
            for from_part in range(3):
                if len(partition[from_part]) == 0:
                    continue
                for to_part in range(3):
                    if from_part == to_part:
                        continue
                    # Try moving each vertex from from_part to to_part
                    for v in list(partition[from_part]):
                        if time.time() > deadline - 15:
                            return partition, current_score
                        
                        # Move v
                        partition[from_part].remove(v)
                        partition[to_part].append(v)
                        
                        new_score = max_tf_size_partition(partition)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            # Revert
                            partition[to_part].remove(v)
                            partition[from_part].append(v)
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return partition, current_score
    
    # Try multiple initial partitions
    initial_partitions = [
        [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))],  # Balanced
        [list(range(0, 5)), list(range(5, 13)), list(range(13, 18))],  # [5,8,5]
        [list(range(0, 4)), list(range(4, 14)), list(range(14, 18))],  # [4,10,4]
        [list(range(0, 3)), list(range(3, 15)), list(range(15, 18))],  # [3,12,3]
        [list(range(0, 2)), list(range(2, 16)), list(range(16, 18))],  # [2,14,2]
        [list(range(0, 1)), list(range(1, 17)), list(range(17, 18))],  # [1,16,1]
    ]
    
    # Also try random partitions
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        random_partition = [[], [], []]
        for v in range(n):
            random_partition[rng.randint(0, 2)].append(v)
        initial_partitions.append(random_partition)
    
    for init_part in initial_partitions:
        if time.time() > deadline - 15:
            break
        
        optimized_part, score = optimize_partition(init_part)
        
        if score < best_score:
            best_score = score
            best_edges = partition_to_edges(optimized_part)
        
        # Additional local search on the edge set for further improvement
        if time.time() > deadline - 10:
            break
        
        # Try removing some edges between parts to potentially reduce TF size
        # This breaks the pure Turán structure but might help
        current_edges = set(best_edges)
        current_adj = build_adj_from_partition(optimized_part)
        current_score = score
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            # Try removing a random edge between parts
            all_inter_part_edges = [(u, v) for u in range(n) for v in range(u + 1, n) 
                                    if optimized_part.index and False]  # placeholder
            # Get actual inter-part edges
            part_of = [0] * n
            for pi, p in enumerate(optimized_part):
                for v in p:
                    part_of[v] = pi
            
            inter_edges = [(u, v) for u in range(n) for v in range(u + 1, n) 
                          if part_of[u] != part_of[v] and (u, v) in current_edges]
            
            if not inter_edges:
                break
            
            u, v = rng.choice(inter_edges)
            current_edges.discard((u, v))
            
            adj = [0] * n
            for a, b in current_edges:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
            
            new_score = max_tf_size_fast(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(current_edges)
            else:
                current_edges.add((u, v))
                no_improve += 1
    
    return [[u, v] for u, v in sorted(best_edges)]
