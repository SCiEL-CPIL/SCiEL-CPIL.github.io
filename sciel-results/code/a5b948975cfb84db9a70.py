"""Direct construction of the Fano plane on vertices 0..6; vertices 7,8 isolated."""

def construct():
    e = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5],
    ]
    c = [0, 1, 2, 0, 1, 0, 2, 0, 0]
    return {'edges': e, 'colors': c}
