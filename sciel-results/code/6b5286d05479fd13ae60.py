<<<<<<< SEARCH
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            idx_u = visited[vertex]
            idx_v = visited[neighbor]
            if idx_u < idx_v:
                edges.append((idx_u, idx_v))
    
    return len(order), edges
=======
        idx_u = visited[vertex]
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            idx_v = visited[neighbor]
            if idx_u < idx_v:
                edges.append((idx_u, idx_v))
    
    return len(order), edges
>>>>>>> REPLACE
