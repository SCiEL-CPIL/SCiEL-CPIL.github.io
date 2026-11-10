def construct():
    es=set()
    for i in range(6):
        for u,v in [(i,(i+1)%6),(i+6,(i+1)%6+6),(i,i+6)]:es.add(tuple(sorted((u,v))))
    return [list(e) for e in sorted(es)]
