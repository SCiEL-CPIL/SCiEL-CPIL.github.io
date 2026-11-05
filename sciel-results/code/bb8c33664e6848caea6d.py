import random
import time
import math
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110
    
    A = np.zeros((n, n), dtype=np.int32)
    cn = np.zeros((n, n), dtype=np.int32)
    
    def update_cn():
        nonlocal cn
        cn = A @ A
    
    def is_valid():
        for i in range(n):
            for j in range(i+1, n):
                if cn[i][j] > 2:
                    return False
        return True
    
    def can_add(i, j):
        if cn[i][j] > 2:
            return False
        for k in range(n):
            if k != i and k != j and A[i][k] and A[j][k]:
                if cn[i][k] >= 2 or cn[j][k] >= 2:
                    return False
        return True
    
    def ec():
        return int(A.sum() // 2)
    
    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    edges.append([i, j])
        return edges
    
    random.seed(42)
    best_edges = 0
    best_A = None
    
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        A[:] = 0
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.2:
                    A[i][j] = A[j][i] = 1
        
        update_cn()
        
        # Fix violations
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    if cn[i][j] > 2:
                        common = [k for k in range(n) if k != i and k != j and A[i][k] and A[j][k]]
                        if common:
                            k = random.choice(common)
                            A[i][k] = A[k][i] = 0
                            update_cn()
                            found = True
                            break
                if found:
                    break
            if not found:
                break
        
        if not is_valid():
            continue
        
        # SA
        temp = 1.0
        for step in range(20000):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old = ec()
            
            if A[i][j]:
                A[i][j] = A[j][i] = 0
                update_cn()
                new = ec()
                if not is_valid():
                    A[i][j] = A[j][i] = 1
                    update_cn()
                    continue
                d = new - old
                if d >= 0 or random.random() < math.exp(d / temp):
                    if new > best_edges:
                        best_edges = new
                        best_A = A.copy()
                else:
                    A[i][j] = A[j][i] = 1
                    update_cn()
            else:
                if not can_add(i, j):
                    continue
                A[i][j] = A[j][i] = 1
                update_cn()
                new = ec()
                if not is_valid():
                    A[i][j] = A[j][i] = 0
                    update_cn()
                    continue
                d = new - old
                if d >= 0 or random.random() < math.exp(d / temp):
                    if new > best_edges:
                        best_edges = new
                        best_A = A.copy()
                else:
                    A[i][j] = A[j][i] = 0
                    update_cn()
            
            temp *= 0.9999
    
    if best_A is not None:
        A[:] = best_A
    
    # Final greedy
    update_cn()
    for i in range(n):
        for j in range(i+1, n):
            if time.time() > deadline - 2:
                break
            if not A[i][j] and can_add(i, j):
                A[i][j] = A[j][i] = 1
                update_cn()
    
    return get_edges()
