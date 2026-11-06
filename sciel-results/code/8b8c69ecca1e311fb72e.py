import os, sys, json
try:
    import autoresearch_suite.science_tcs_expansion as M
    print("MODULE FILE:", os.path.abspath(M.__file__))
    print("NAMES:", [n for n in dir(M) if not n.startswith('__')])
except Exception as e:
    print("IMPORT ERR", e)
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for f in files:
            if f.endswith('.py'):
                print(os.path.join(root, f))
