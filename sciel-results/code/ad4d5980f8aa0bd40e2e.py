import time
import numpy as np
from itertools import combinations
import random

def construct():
    N = 40
    M = 401
    deadline = time.time() + 110
    
    all_offsets = list(range(1, 20))
    
    best_edges = None
    best_score = float('inf')
    best_adj = None
    best_non_edges = None

    def compute_books(adj, edges):
        books = {}
        max_b = 0
        for i, j in edges:
            b = (adj[i] & adj[j]).bit_count()
            books[(i,j)] = b
            if b > max_b:
                max_b = b
        return max_b, books

    def local_search(adj, edges, non_edges, current_max_book, time_limit=10.0):
        start_time = time.time()
        edge_set = set(edges)
        
        _, books = compute_books(adj, edges)
        
        while time.time() < start_time + time_limit and time.time() < deadline - 5:
            max_edges = [e for e in edges if books.get(e, 0) == current_max_book]
            
            if not max_edges:
                break
                
            improved = False
            random.shuffle(max_edges)
            
            for e_remove in max_edges:
                u, v = e_remove
                if len(non_edges) < 20:
                    candidates = non_edges[:]
                else:
                    candidates = random.sample(non_edges, 20)
                
                for e_add in candidates:
                    x, y = e_add
                    
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    
                    new_edges = [e for e in edges if e != e_remove]
                    new_edges.append(e_add)
                    
                    new_max, new_books = compute_books(adj, new_edges)
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
                    
                    if new_max < current_max_book:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        
                        edges.remove(e_remove)
                        edges.append(e_add)
                        non_edges.remove(e_add)
                        non_edges.append(e_remove)
                        
                        current_max_book = new_max
                        books = new_books
                        improved = True
                        break
                    elif new_max == current_max_book:
                        old_max_count = sum(1 for b in books.values() if b == current_max_book)
                        new_max_count = sum(1 for b in new_books.values() if b == new_max)
                        if new_max_count < old_max_count:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            adj[x] |= (1 << y)
                            adj[y] |= (1 << x)
                            
                            edges.remove(e_remove)
                            edges.append(e_add)
                            non_edges.remove(e_add)
                            non_edges.append(e_remove)
                            
                            books = new_books
                            improved = True
                            break

            if not improved:
                pass

        return adj, edges, non_edges, current_max_book

    for S in combinations(all_offsets, 10):
        if time.time() > deadline - 30:
            break
            
        S_set = set(S)
        
        max_circ_book = 0
        for d in S:
            count = 0
            for s in S:
                if (s - d) % 40 in S_set:
                    count += 1
            if count > max_circ_book:
                max_circ_book = count
                
        if max_circ_book > best_score:
            continue
            
        adj = [0] * N
        for i in range(N):
            mask = 0
            for s in S:
                j1 = (i + s) % N
                j2 = (i - s) % N
                mask |= (1 << j1)
                mask |= (1 << j2)
            adj[i] = mask
            
        edges = []
        edge_set = set()
        for i in range(N):
            mask = adj[i]
            for j in range(i+1, N):
                if (mask >> j) & 1:
                    edges.append((i, j))
                    edge_set.add((i, j))
                    
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        base_books = {}
        max_base_book = 0
        for i, j in edges:
            b = (adj[i] & adj[j]).bit_count()
            base_books[(i,j)] = b
            if b > max_base_book:
                max_base_book = b
                
        best_candidate_score = float('inf')
        best_candidate_edge = None
        
        for u, v in non_edges:
            common_mask = adj[u] & adj[v]
            b_new = common_mask.bit_count()
            
            current_max = max_base_book
            if b_new > current_max:
                current_max = b_new
                
            cm = common_mask
            while cm:
                lsb = cm & -cm
                a = lsb.bit_length() - 1
                cm ^= lsb
                
                ua = (u, a) if u < a else (a, u)
                va = (v, a) if v < a else (a, v)
                
                bua = base_books.get(ua, 0)
                bva = base_books.get(va, 0)
                
                if bua + 1 > current_max:
                    current_max = bua + 1
                if bva + 1 > current_max:
                    current_max = bva + 1
                    
            if current_max < best_candidate_score:
                best_candidate_score = current_max
                best_candidate_edge = (u, v)
                if best_candidate_score == 7:
                    break
                    
        if best_candidate_score < best_score:
            best_score = best_candidate_score
            best_edges = [list(e) for e in edges]
            best_edges.append(list(best_candidate_edge))
            best_adj = adj[:]
            best_non_edges = [e for e in non_edges if e != best_candidate_edge]
            
            if best_score == 7:
                return best_edges

    if best_edges and best_adj:
        adj_ls = [0] * N
        for u, v in best_edges:
            adj_ls[u] |= (1 << v)
            adj_ls[v] |= (1 << u)
            
        edges_ls = [tuple(e) for e in best_edges]
        edge_set_ls = set(edges_ls)
        all_pairs_ls = [(i, j) for i in range(N) for j in range(i+1, N)]
        non_edges_ls = [p for p in all_pairs_ls if p not in edge_set_ls]
        
        remaining = deadline - time.time() - 5
        if remaining > 10:
            adj_ls, edges_ls, non_edges_ls, final_score = local_search(
                adj_ls, edges_ls, non_edges_ls, best_score, time_limit=remaining
            )
            if final_score < best_score:
                best_score = final_score
                best_edges = [list(e) for e in edges_ls]
        
        return best_edges

    random_edges = []
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(pairs)
    for u, v in pairs[:M]:
        random_edges.append([u, v])
    return random_edges
