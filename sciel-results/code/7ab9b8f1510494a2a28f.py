import time
import sys
import random

def construct():
    start_time = time.time()
    timeout = 115
    
    max_n = 35
    
    adj = [set() for _ in range(max_n)]
    indep = [set() for _ in range(6)]
    triangles = []
    
    # Initialize
    indep[1].add((0,))
    current_n = 1
    
    def dpll(clauses, assignment, depth=0):
        if time.time() - start_time > timeout:
            return None
            
        if depth > 50: # Prevent infinite recursion
            return None
            
        new_clauses = []
        for clause in clauses:
            satisfied = False
            new_clause = []
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    val = assignment[var]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                else:
                    new_clause.append(lit)
            if satisfied:
                continue
            if not new_clause:
                return None
            if len(new_clause) == 1:
                lit = new_clause[0]
                var = abs(lit)
                val = lit > 0
                if var in assignment and assignment[var] != val:
                    return None
                new_assignment = dict(assignment)
                new_assignment[var] = val
                result = dpll(clauses, new_assignment, depth + 1)
                if result is not None:
                    return result
                else:
                    return None
            else:
                new_clauses.append(tuple(new_clause))
                
        if not new_clauses:
            return assignment
            
        var = abs(new_clauses[0][0])
        
        assignment[var] = True
        result = dpll(new_clauses, assignment, depth + 1)
        if result is not None:
            return result
            
        assignment[var] = False
        result = dpll(new_clauses, assignment, depth + 1)
        if result is not None:
            return result
            
        del assignment[var]
        return None

    def add_vertex(v):
        clauses = []
        
        for a, b, c in triangles:
            clauses.append((-a, -b, -c))
            
        for s in indep[5]:
            a, b, c, d, e = s
            clauses.append((a, b, c, d, e))
            
        # Subsumption: remove clauses that are supersets of others
        # This is O(N^2) but N is small
        # Sort by length
        clauses = [tuple(sorted(c)) for c in clauses]
        clauses = sorted(clauses, key=len)
        
        filtered_clauses = []
        for c in clauses:
            redundant = False
            for fc in filtered_clauses:
                if len(fc) <= len(c):
                    if set(fc).issubset(set(c)):
                        redundant = True
                        break
            if not redundant:
                filtered_clauses.append(c)
                
        solution = dpll(filtered_clauses, {})
        
        if solution is None:
            return False
        
        for u in range(v):
            if solution.get(u, False):
                adj[v].add(u)
                adj[u].add(v)
                
        # Update Triangles
        for a in adj[v]:
            for b in adj[v]:
                if a < b and b in adj[a]:
                    triangles.append((a, b, v))
                    
        # Update Independent Sets
        indep[1].add((v,))
        
        for u in range(v):
            if u not in adj[v]:
                indep[2].add((u, v))
                
        for s in list(indep[2]):
            a, b = s
            if a != v and b != v:
                if a not in adj[v] and b not in adj[v]:
                    indep[3].add(tuple(sorted((a, b, v))))
                    
        for s in list(indep[3]):
            a, b, c = s
            if v not in s:
                if a not in adj[v] and b not in adj[v] and c not in adj[v]:
                    indep[4].add(tuple(sorted((a, b, c, v))))
                    
        for s in list(indep[4]):
            a, b, c, d = s
            if v not in s:
                if a not in adj[v] and b not in adj[v] and c not in adj[v] and d not in adj[v]:
                    indep[5].add(tuple(sorted((a, b, c, d, v))))
                    
        return True

    for v in range(1, max_n):
        if time.time() - start_time > timeout:
            break
        if add_vertex(v):
            current_n = v + 1
        else:
            break
            
    edges = []
    for u in range(current_n):
        for v in range(u + 1, current_n):
            if v in adj[u]:
                edges.append([u, v])
                
    if current_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
        
    return {'n': current_n, 'edges': edges}
