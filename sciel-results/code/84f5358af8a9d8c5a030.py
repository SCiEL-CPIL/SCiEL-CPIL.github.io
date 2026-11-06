<<<<<<< SEARCH
    def try_extend(path, adj):
        """Try to extend path at both ends."""
        path = list(path)
        changed = True
        while changed:
            changed = False
            # Try to extend at the front
            front = path[0]
            for v in sorted(adj[front]):
                if v not in set(path):
                    path.insert(0, v)
                    changed = True
                    break
            # Try to extend at the back
            if not changed:
                back = path[-1]
                for v in sorted(adj[back]):
                    if v not in set(path):
                        path.append(v)
                        changed = True
                        break
        return path
=======
    def try_extend(path, adj):
        """Try to extend path at both ends."""
        path = list(path)
        visited = set(path)
        changed = True
        while changed:
            changed = False
            # Try to extend at the front
            front = path[0]
            for v in sorted(adj[front]):
                if v not in visited:
                    path.insert(0, v)
                    visited.add(v)
                    changed = True
                    break
            # Try to extend at the back
            if not changed:
                back = path[-1]
                for v in sorted(adj[back]):
                    if v not in visited:
                        path.append(v)
                        visited.add(v)
                        changed = True
                        break
        return path
>>>>>>> REPLACE

