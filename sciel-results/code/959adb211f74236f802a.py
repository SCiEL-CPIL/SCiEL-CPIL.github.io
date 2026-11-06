import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    N = 20
    C = 20
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def get_4cycles_containing_edge(adj, u, v):
        """Get all 4-cycles that use edge (u,v). Each 4-cycle is (u,v,x,y) meaning u-v-x-y-u."""
        cycles = []
        for x in range(N):
            if x == u or x == v:
                continue
            if (min(v,x), max(v,x)) not in adj:
                continue
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                if (min(u,y), max(u,y)) not in adj:
                    continue
                if (min(x,y), max(x,y)) not in adj:
                    continue
                cycles.append((u, v, x, y))
        return cycles
    
    def is_addition_valid(adj, u, v, c):
        key = (min(u,v), max(u,v))
        if key in adj:
            return False
        for (a, b, x, y) in get_4cycles_containing_edge(adj, u, v):
            # Cycle: a-b-x-y-a, new edge is (a,b) with colour c
            cb = adj[(min(b,x), max(b,x))]
            cc = adj[(min(x,y), max(x,y))]
            cd = adj[(min(y,a), max(y,a))]
            if len({c, cb, cc, cd}) != 4:
                return False
        return True
    
    def best_colour(adj, u, v):
        key = (min(u,v), max(u,v))
        if key in adj:
            return None
        best_c = None
        best_score = float('inf')
        for c in range(C):
            if is_addition_valid(adj, u, v, c):
                # Score: count of 4-cycles where c clashes with existing colours (should be 0 if valid)
                # Prefer colours that appear less often in the graph
                score = 0
                for (a, b, x, y) in get_4cycles_containing_edge(adj, u, v):
                    cb = adj[(min(b,x), max(b,x))]
                    cc = adj[(min(x,y), max(x,y))]
                    cd = adj[(min(y,a), max(y,a))]
                    if c == cb or c == cc or c == cd:
                        score += 1
                # Also penalise if colour c is already used a lot
                if score < best_score:
                    best_score = score
                    best_c = c
        return best_c
    
    def greedy():
        adj = {}
        edges = [(i,j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(edges)
        for (u,v) in edges:
            if time_left() < 5.0:
                break
            c = best_colour(adj, u, v)
            if c is not None:
                adj[(min(u,v), max(u,v))] = c
        return adj
    
    def local_search(adj):
        improved = True
        while improved and time_left() > 15.0:
            improved = False
            # Add edges
            missing = [(i,j) for i in range(N) for j in range(i+1, N) if (min(i,j),max(i,j)) not in adj]
            random.shuffle(missing)
            for (u,v) in missing:
                if time_left() < 15.0:
                    break
                c = best_colour(adj, u, v)
                if c is not None:
                    adj[(min(u,v), max(u,v))] = c
                    improved = True
            # Recolour to enable more additions
            for (u,v), c in list(adj.items()):
                if time_left() < 15.0:
                    break
                for nc in range(C):
                    if nc == c:
                        continue
                    del adj[(u,v)]
                    if is_addition_valid(adj, u, v, nc):
                        adj[(u,v)] = nc
                        improved = True
                        # Try adding edges after recolor
                        missing2 = [(i,j) for i in range(N) for j in range(i+1, N) if (min(i,j),max(i,j)) not in adj]
                        random.shuffle(missing2)
                        for (a,b) in missing2:
                            if time_left() < 15.0:
                                break
                            c2 = best_colour(adj, a, b)
                            if c2 is not None:
                                adj[(min(a,b), max(a,b))] = c2
                                improved = True
                        break
                    else:
                        adj[(u,v)] = c
        return adj
    
    best_adj = None
    best_count = 0
    
    for trial in range(15):
        if time_left() < 25.0:
            break
        random.seed(1000 + trial)
        adj = greedy()
        adj = local_search(adj)
        if len(adj) > best_count:
            best_count = len(adj)
            best_adj = dict(adj)
    
    if best_adj is not None and time_left() > 10.0:
        best_adj = local_search(best_adj)
        best_count = len(best_adj)
    
    if best_adj is None:
        return [[0, v, 0] for v in range(1, 20)]
    
    return [[u, v, c] for (u, v), c in best_adj.items()]
