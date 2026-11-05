import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 40
    m = 401
    target_max_book = 7

    def build_adj(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def book_size(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def compute_books(edge_set, adj):
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = book_size(u, v, adj)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def greedy_construct(target_edges, max_book_limit):
        edge_set = set()
        adj = [0] * n
        books = {}
        
        for step in range(target_edges):
            if time.time() > deadline - 5:
                break
                
            best_edge = None
            best_b = max_book_limit + 1
            
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in edge_set:
                        continue
                    
                    common_mask = adj[u] & adj[v]
                    b_uv = bin(common_mask).count('1')
                    
                    valid = True
                    if common_mask:
                        for w in range(n):
                            if (common_mask >> w) & 1:
                                if u < w:
                                    b_uw = books.get((u, w), 0)
                                else:
                                    b_uw = books.get((w, u), 0)
                                    
                                if b_uw >= max_book_limit:
                                    valid = False
                                    break
                                    
                                if v < w:
                                    b_vw = books.get((v, w), 0)
                                else:
                                    b_vw = books.get((w, v), 0)
                                    
                                if b_vw >= max_book_limit:
                                    valid = False
                                    break
                            
                    if valid:
                        if b_uv < best_b:
                            best_b = b_uv
                            best_edge = (u, v)
                            
            if best_edge is None:
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) in edge_set:
                            continue
                        common_mask = adj[u] & adj[v]
                        b_uv = bin(common_mask).count('1')
                        if b_uv < best_b:
                            best_b = b_uv
                            best_edge = (u, v)
                            
            if best_edge is None:
                break
                
            u, v = best_edge
            
            edge_set.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            books[(u, v)] = best_b
            
            common_mask = adj[u] & adj[v]
            
            for w in range(n):
                if (common_mask >> w) & 1:
                    if u < w:
                        books[(u, w)] = books.get((u, w), 0) + 1
                    else:
                        books[(w, u)] = books.get((w, u), 0) + 1
                        
                    if v < w:
                        books[(v, w)] = books.get((v, w), 0) + 1
                    else:
                        books[(w, v)] = books.get((w, v), 0) + 1

        return edge_set, books, adj

    def local_search(edge_set, books, adj, max_steps=5000):
        current_score = max(books.values()) if books else 0
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e
                    
            if worst_val <= target_max_book:
                if len(edge_set) < m:
                    added = False
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (u, v) in edge_set:
                                continue
                            common_mask = adj[u] & adj[v]
                            b_uv = bin(common_mask).count('1')
                            valid = True
                            if common_mask:
                                for w in range(n):
                                    if (common_mask >> w) & 1:
                                        if u < w:
                                            b_uw = books.get((u, w), 0)
                                        else:
                                            b_uw = books.get((w, u), 0)
                                        if b_uw >= target_max_book:
                                            valid = False
                                            break
                                        if v < w:
                                            b_vw = books.get((v, w), 0)
                                        else:
                                            b_vw = books.get((w, v), 0)
                                        if b_vw >= target_max_book:
                                            valid = False
                                            break
                            if valid and b_uv <= target_max_book:
                                edge_set.add((u, v))
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                books[(u, v)] = b_uv
                                for w in range(n):
                                    if (common_mask >> w) & 1:
                                        if u < w:
                                            books[(u, w)] = books.get((u, w), 0) + 1
                                        else:
                                            books[(w, u)] = books.get((w, u), 0) + 1
                                        if v < w:
                                            books[(v, w)] = books.get((v, w), 0) + 1
                                        else:
                                            books[(w, v)] = books.get((w, v), 0) + 1
                                added = True
                                break
                        if added:
                            break
                    if added:
                        continue
                    else:
                        break
                else:
                    break
                
            u, v = worst_edge
            found_swap = False
            for _ in range(200):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) in edge_set:
                    continue
                if (ru, rv) == (u, v):
                    continue
                    
                edge_set.remove((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                del books[(u, v)]
                
                common_mask_uv = adj[u] & adj[v]
                for w in range(n):
                    if (common_mask_uv >> w) & 1:
                        if u < w:
                            books[(u, w)] -= 1
                        else:
                            books[(w, u)] -= 1
                        if v < w:
                            books[(v, w)] -= 1
                        else:
                            books[(w, v)] -= 1
                            
                edge_set.add((ru, rv))
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                
                common_mask_ru = adj[ru] & adj[rv]
                b_ru = bin(common_mask_ru).count('1')
                books[(ru, rv)] = b_ru
                
                for w in range(n):
                    if (common_mask_ru >> w) & 1:
                        if ru < w:
                            books[(ru, w)] = books.get((ru, w), 0) + 1
                        else:
                            books[(w, ru)] = books.get((w, ru), 0) + 1
                        if rv < w:
                            books[(rv, w)] = books.get((rv, w), 0) + 1
                        else:
                            books[(w, rv)] = books.get((w, rv), 0) + 1
                            
                new_max = max(books.values())
                
                if new_max < current_score:
                    current_score = new_max
                    found_swap = True
                    break
                else:
                    edge_set.remove((ru, rv))
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    del books[(ru, rv)]
                    
                    for w in range(n):
                        if (common_mask_ru >> w) & 1:
                            if ru < w:
                                books[(ru, w)] -= 1
                            else:
                                books[(w, ru)] -= 1
                            if rv < w:
                                books[(rv, w)] -= 1
                            else:
                                books[(w, rv)] -= 1
                                
                    edge_set.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    books[(u, v)] = worst_val
                    
                    for w in range(n):
                        if (common_mask_uv >> w) & 1:
                            if u < w:
                                books[(u, w)] += 1
                            else:
                                books[(w, u)] += 1
                            if v < w:
                                books[(v, w)] += 1
                            else:
                                books[(w, v)] += 1
                                
            if not found_swap:
                if current_score <= target_max_book and len(edge_set) == m:
                    break
                pass
                    
        return current_score

    edge_set, books, adj = greedy_construct(m, target_max_book)
    
    if len(edge_set) < m or max(books.values()) > target_max_book:
        local_search(edge_set, books, adj)
        
    return [[u, v] for (u, v) in edge_set]
