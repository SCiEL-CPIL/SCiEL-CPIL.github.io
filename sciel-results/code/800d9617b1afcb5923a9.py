    def creates_c6(u, v):
        stack = [(u, [u])]
        while stack:
            node, path = stack.pop()
            if len(path) == 5:
                if v in adj[node]:
                    return True
                continue
            for nei in adj[node]:
                if nei not in path:
                    stack.append((nei, path + [nei]))
        return False
